"""CDK stack for Sunday Album Web UI — Phase 1: Infrastructure + Auth + Upload."""

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigatewayv2 as apigw,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
)
from aws_cdk.aws_apigatewayv2_integrations import HttpLambdaIntegration
from constructs import Construct


class SundayAlbumStack(Stack):
    """All Sunday Album AWS resources in a single stack."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── S3 Bucket ────────────────────────────────────────────────────────
        bucket = s3.Bucket(
            self,
            "DataBucket",
            bucket_name=f"sundayalbum-data-{self.account}-{self.region}",
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-uploads",
                    prefix="uploads/",
                    expiration=Duration.days(30),
                ),
                s3.LifecycleRule(
                    id="expire-debug",
                    prefix="debug/",
                    expiration=Duration.days(7),
                ),
                s3.LifecycleRule(
                    id="expire-output",
                    prefix="output/",
                    expiration=Duration.days(30),
                ),
            ],
            cors=[
                s3.CorsRule(
                    allowed_methods=[
                        s3.HttpMethods.GET,
                        s3.HttpMethods.PUT,
                        s3.HttpMethods.HEAD,
                    ],
                    allowed_origins=["*"],  # restrict in Phase 6
                    allowed_headers=["*"],
                    max_age=3000,
                )
            ],
        )

        # ── DynamoDB Tables ──────────────────────────────────────────────────
        # sa-sessions: PK=email, GSI on session_token for auth middleware
        sessions_table = dynamodb.Table(
            self,
            "SessionsTable",
            table_name="sa-sessions",
            partition_key=dynamodb.Attribute(
                name="email", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="token_expires_at",
            removal_policy=RemovalPolicy.RETAIN,
        )
        sessions_table.add_global_secondary_index(
            index_name="token-index",
            partition_key=dynamodb.Attribute(
                name="session_token", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # sa-jobs: PK=user_hash, SK=job_id (ULID) — DynamoDB Stream for Phase 3
        jobs_table = dynamodb.Table(
            self,
            "JobsTable",
            table_name="sa-jobs",
            partition_key=dynamodb.Attribute(
                name="user_hash", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="job_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.RETAIN,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
        )

        # sa-ws-connections: PK=connection_id (used in Phase 3)
        ws_table = dynamodb.Table(
            self,
            "WsConnectionsTable",
            table_name="sa-ws-connections",
            partition_key=dynamodb.Attribute(
                name="connection_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )
        ws_table.add_global_secondary_index(
            index_name="job-index",
            partition_key=dynamodb.Attribute(
                name="job_id", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # ── Shared Lambda execution role ─────────────────────────────────────
        ses_sender = (
            self.node.try_get_context("ses_sender_email") or "noreply@sundayalbum.com"
        )

        lambda_role = iam.Role(
            self,
            "ApiLambdaRole",
            role_name="sa-api-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # DynamoDB access
        sessions_table.grant_read_write_data(lambda_role)
        jobs_table.grant_read_write_data(lambda_role)
        ws_table.grant_read_write_data(lambda_role)

        # S3 access (read/write + presigned URL generation)
        bucket.grant_read_write(lambda_role)

        # SES: send emails
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="SesEmailSend",
                actions=["ses:SendEmail", "ses:SendRawEmail"],
                resources=["*"],
            )
        )

        # Step Functions: start executions (state machine ARN added in Phase 2)
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="StepFunctionsStart",
                actions=["states:StartExecution", "states:DescribeExecution"],
                resources=["*"],
            )
        )

        # ── Shared environment variables ─────────────────────────────────────
        common_env = {
            "SESSIONS_TABLE": sessions_table.table_name,
            "JOBS_TABLE": jobs_table.table_name,
            "WS_CONNECTIONS_TABLE": ws_table.table_name,
            "S3_BUCKET": bucket.bucket_name,
            "SES_SENDER": ses_sender,
            "AWS_ACCOUNT_ID": self.account,
            "AWS_DEPLOY_REGION": self.region,
        }

        # ── Lambda functions ─────────────────────────────────────────────────
        # All three share the same code asset (api/ directory); different handler
        lambda_code = lambda_.Code.from_asset("../api")
        py312 = lambda_.Runtime.PYTHON_3_12

        auth_fn = lambda_.Function(
            self,
            "AuthFunction",
            function_name="sa-auth",
            runtime=py312,
            handler="auth.handler",
            code=lambda_code,
            role=lambda_role,
            environment=common_env,
            timeout=Duration.seconds(30),
            memory_size=512,
            description="Sunday Album auth: send-code, verify, logout",
        )

        jobs_fn = lambda_.Function(
            self,
            "JobsFunction",
            function_name="sa-jobs",
            runtime=py312,
            handler="jobs.handler",
            code=lambda_code,
            role=lambda_role,
            environment=common_env,
            timeout=Duration.seconds(30),
            memory_size=512,
            description="Sunday Album jobs: CRUD, presigned URLs, pipeline trigger",
        )

        ws_fn = lambda_.Function(
            self,
            "WebSocketFunction",
            function_name="sa-websocket",
            runtime=py312,
            handler="websocket.handler",
            code=lambda_code,
            role=lambda_role,
            environment=common_env,
            timeout=Duration.seconds(29),
            memory_size=256,
            description="Sunday Album WebSocket: connect, disconnect, broadcast",
        )

        # ── API Gateway HTTP API ─────────────────────────────────────────────
        http_api = apigw.HttpApi(
            self,
            "HttpApi",
            api_name="sundayalbum-api",
            cors_preflight=apigw.CorsPreflightOptions(
                allow_origins=["*"],  # restrict in Phase 6
                allow_methods=[apigw.CorsHttpMethod.ANY],
                allow_headers=["Content-Type", "Authorization"],
                max_age=Duration.hours(1),
            ),
        )

        # Auth routes → sa-auth Lambda
        auth_int = HttpLambdaIntegration("AuthSendCode", auth_fn)
        http_api.add_routes(
            path="/auth/send-code",
            methods=[apigw.HttpMethod.POST],
            integration=auth_int,
        )
        http_api.add_routes(
            path="/auth/verify",
            methods=[apigw.HttpMethod.POST],
            integration=HttpLambdaIntegration("AuthVerify", auth_fn),
        )
        http_api.add_routes(
            path="/auth/logout",
            methods=[apigw.HttpMethod.POST],
            integration=HttpLambdaIntegration("AuthLogout", auth_fn),
        )

        # Jobs routes → sa-jobs Lambda
        http_api.add_routes(
            path="/jobs",
            methods=[apigw.HttpMethod.GET],
            integration=HttpLambdaIntegration("JobsList", jobs_fn),
        )
        http_api.add_routes(
            path="/jobs",
            methods=[apigw.HttpMethod.POST],
            integration=HttpLambdaIntegration("JobsCreate", jobs_fn),
        )
        http_api.add_routes(
            path="/jobs/{jobId}",
            methods=[apigw.HttpMethod.GET],
            integration=HttpLambdaIntegration("JobsGet", jobs_fn),
        )
        http_api.add_routes(
            path="/jobs/{jobId}",
            methods=[apigw.HttpMethod.DELETE],
            integration=HttpLambdaIntegration("JobsDelete", jobs_fn),
        )
        http_api.add_routes(
            path="/jobs/{jobId}/start",
            methods=[apigw.HttpMethod.POST],
            integration=HttpLambdaIntegration("JobsStart", jobs_fn),
        )
        http_api.add_routes(
            path="/jobs/{jobId}/reprocess",
            methods=[apigw.HttpMethod.POST],
            integration=HttpLambdaIntegration("JobsReprocess", jobs_fn),
        )

        # ── Outputs ──────────────────────────────────────────────────────────
        CfnOutput(
            self,
            "ApiUrl",
            value=http_api.api_endpoint,
            export_name="SundayAlbumApiUrl",
            description="HTTP API base URL",
        )
        CfnOutput(
            self,
            "BucketName",
            value=bucket.bucket_name,
            export_name="SundayAlbumBucketName",
            description="S3 data bucket",
        )
        CfnOutput(
            self,
            "SessionsTableName",
            value=sessions_table.table_name,
            export_name="SundayAlbumSessionsTable",
        )
        CfnOutput(
            self,
            "JobsTableName",
            value=jobs_table.table_name,
            export_name="SundayAlbumJobsTable",
        )
        CfnOutput(
            self,
            "AuthFunctionName",
            value=auth_fn.function_name,
            export_name="SundayAlbumAuthFunction",
        )
        CfnOutput(
            self,
            "JobsFunctionName",
            value=jobs_fn.function_name,
            export_name="SundayAlbumJobsFunction",
        )

        # Store key resource references for cross-stack use in later phases
        self.bucket = bucket
        self.sessions_table = sessions_table
        self.jobs_table = jobs_table
        self.ws_table = ws_table
        self.http_api = http_api
        self.auth_fn = auth_fn
        self.jobs_fn = jobs_fn
        self.ws_fn = ws_fn
