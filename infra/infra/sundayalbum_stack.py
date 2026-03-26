"""CDK stack for Sunday Album Web UI — Phase 1 + Phase 2."""

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigatewayv2 as apigw,
    aws_dynamodb as dynamodb,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
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

        # ── Shared Lambda execution role (API + pipeline) ────────────────────
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

        # Step Functions: start executions (state machine ARN wired below)
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

        # ── API Lambda functions (zip-based) ─────────────────────────────────
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

        # ── Pipeline Lambda functions (container image) ───────────────────────
        # One image, all 11 step handlers. CMD overridden per function via
        # DockerImageCode.from_image_asset(cmd=...).
        # CDK deduplicates the Docker build when directory + platform are identical.
        pipeline_env = {
            "JOBS_TABLE": jobs_table.table_name,
            "S3_BUCKET": bucket.bucket_name,
            "AWS_DEPLOY_REGION": self.region,
            # API keys come from Lambda environment (set manually or via Secrets Manager)
            # ANTHROPIC_API_KEY and OPENAI_API_KEY must be set post-deploy
        }

        def pipeline_fn(
            construct_id: str,
            fn_name: str,
            handler: str,
            timeout_secs: int = 300,
            memory_mb: int = 1024,
            description: str = "",
        ) -> lambda_.DockerImageFunction:
            return lambda_.DockerImageFunction(
                self,
                construct_id,
                function_name=fn_name,
                code=lambda_.DockerImageCode.from_image_asset(
                    "../",  # build context = repo root (Dockerfile is there)
                    cmd=[handler],
                    platform=ecr_assets.Platform.LINUX_ARM64,
                ),
                role=lambda_role,
                environment=pipeline_env,
                timeout=Duration.seconds(timeout_secs),
                memory_size=memory_mb,
                architecture=lambda_.Architecture.ARM_64,
                description=description or fn_name,
            )

        load_fn = pipeline_fn(
            "LoadFunction", "sa-pipeline-load", "handlers.load.handler",
            timeout_secs=60, memory_mb=3008, description="Load & decode source image",
        )
        normalize_fn = pipeline_fn(
            "NormalizeFunction", "sa-pipeline-normalize", "handlers.normalize.handler",
            timeout_secs=60, memory_mb=3008, description="Resize & orient image",
        )
        page_detect_fn = pipeline_fn(
            "PageDetectFunction", "sa-pipeline-page-detect", "handlers.page_detect.handler",
            timeout_secs=120, memory_mb=3008, description="Detect album page boundary",
        )
        perspective_fn = pipeline_fn(
            "PerspectiveFunction", "sa-pipeline-perspective", "handlers.perspective.handler",
            timeout_secs=120, memory_mb=3008, description="Perspective / keystone correction",
        )
        photo_detect_fn = pipeline_fn(
            "PhotoDetectFunction", "sa-pipeline-photo-detect", "handlers.photo_detect.handler",
            timeout_secs=120, memory_mb=3008, description="Detect individual photo boundaries",
        )
        photo_split_fn = pipeline_fn(
            "PhotoSplitFunction", "sa-pipeline-photo-split", "handlers.photo_split.handler",
            timeout_secs=120, memory_mb=3008, description="Extract individual photo crops",
        )
        ai_orient_fn = pipeline_fn(
            "AiOrientFunction", "sa-pipeline-ai-orient", "handlers.ai_orient.handler",
            timeout_secs=60, memory_mb=1024, description="AI orientation correction (Claude Haiku)",
        )
        glare_remove_fn = pipeline_fn(
            "GlareRemoveFunction", "sa-pipeline-glare-remove", "handlers.glare_remove.handler",
            timeout_secs=300, memory_mb=3008, description="Glare removal (OpenAI gpt-image-1.5)",
        )
        geometry_fn = pipeline_fn(
            "GeometryFunction", "sa-pipeline-geometry", "handlers.geometry.handler",
            timeout_secs=120, memory_mb=3008, description="Geometry correction (dewarp, rotation)",
        )
        color_restore_fn = pipeline_fn(
            "ColorRestoreFunction", "sa-pipeline-color-restore", "handlers.color_restore.handler",
            timeout_secs=120, memory_mb=3008, description="Color restoration (WB, deyellow, CLAHE, sharpen)",
        )
        finalize_fn = pipeline_fn(
            "FinalizeFunction", "sa-pipeline-finalize", "handlers.finalize.handler",
            timeout_secs=60, memory_mb=512, description="Collect results, mark job complete",
        )

        # ── Step Functions State Machine ─────────────────────────────────────
        # Helper: wrap a Lambda in a LambdaInvoke task (passes full state through).
        #
        # Step Functions Lambda:Invoke wraps the result:
        #   {"StatusCode": 200, "Payload": {...handler_return...}}
        # Using output_path="$.Payload" extracts just the handler's return value
        # as the effective output sent to the next state — no wrapper key.
        def invoke(task_id: str, fn: lambda_.IFunction, comment: str = "") -> sfn_tasks.LambdaInvoke:
            return sfn_tasks.LambdaInvoke(
                self,
                task_id,
                lambda_function=fn,
                payload=sfn.TaskInput.from_json_path_at("$"),
                output_path="$.Payload",
                comment=comment or task_id,
            )

        # Sequential steps (whole-page)
        load_task = invoke("Load", load_fn, "Decode source image from S3")
        normalize_task = invoke("Normalize", normalize_fn, "Resize & fix orientation")
        page_detect_task = invoke("PageDetect", page_detect_fn, "Detect album page boundary")
        perspective_task = invoke("Perspective", perspective_fn, "Apply perspective correction")
        photo_detect_task = invoke("PhotoDetect", photo_detect_fn, "Detect individual photo regions")
        photo_split_task = invoke("PhotoSplit", photo_split_fn, "Extract individual photo crops")

        # Build per-photo task array: [1, 2, ..., photo_count]
        # States.ArrayRange(1, photo_count, 1) generates [1, 2, ..., N]
        prepare_map = sfn.Pass(
            self,
            "PrepareMap",
            comment="Generate photo index array for Map state",
            parameters={
                "user_hash.$": "$.user_hash",
                "job_id.$": "$.job_id",
                "stem.$": "$.stem",
                "start_time.$": "$.start_time",
                "photo_count.$": "$.photo_count",
                "config.$": "$.config",
                # Array of integers [1..photo_count] for Map state iterator
                "photo_indices.$": "States.ArrayRange(1, $.photo_count, 1)",
            },
        )

        # Per-photo pipeline (runs inside Map state)
        ai_orient_task = invoke("AiOrient", ai_orient_fn, "AI orientation correction")
        glare_remove_task = invoke("GlareRemove", glare_remove_fn, "Glare removal")
        geometry_task = invoke("Geometry", geometry_fn, "Geometry correction")
        color_restore_task = invoke("ColorRestore", color_restore_fn, "Color restoration")

        per_photo_chain = (
            ai_orient_task
            .next(glare_remove_task)
            .next(geometry_task)
            .next(color_restore_task)
        )

        # Map state: iterate over photo_indices, run per-photo chain in parallel
        # Each iteration receives {user_hash, job_id, stem, start_time, config, photo_index}
        process_photos_map = sfn.Map(
            self,
            "ProcessPhotos",
            comment="Process each photo in parallel (AI orient → glare → geometry → color)",
            items_path="$.photo_indices",
            item_selector={
                # $$.Map.Item.Value = current photo index integer (1, 2, 3 ...)
                "photo_index.$": "$$.Map.Item.Value",
                # $ = parent state (PrepareMap output)
                "user_hash.$": "$.user_hash",
                "job_id.$": "$.job_id",
                "stem.$": "$.stem",
                "start_time.$": "$.start_time",
                "config.$": "$.config",
            },
            max_concurrency=4,
            result_path="$.photo_results",
        )
        process_photos_map.item_processor(per_photo_chain)

        finalize_task = invoke("Finalize", finalize_fn, "Collect results, mark job complete")

        # Chain everything together
        definition = (
            load_task
            .next(normalize_task)
            .next(page_detect_task)
            .next(perspective_task)
            .next(photo_detect_task)
            .next(photo_split_task)
            .next(prepare_map)
            .next(process_photos_map)
            .next(finalize_task)
        )

        # Step Functions execution role
        sfn_role = iam.Role(
            self,
            "StateMachineRole",
            role_name="sa-state-machine-role",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaRole"
                )
            ],
        )
        # Allow state machine to invoke all pipeline Lambdas
        for fn in [
            load_fn, normalize_fn, page_detect_fn, perspective_fn,
            photo_detect_fn, photo_split_fn, ai_orient_fn, glare_remove_fn,
            geometry_fn, color_restore_fn, finalize_fn,
        ]:
            fn.grant_invoke(sfn_role)

        state_machine = sfn.StateMachine(
            self,
            "PipelineStateMachine",
            state_machine_name="sa-pipeline",
            state_machine_type=sfn.StateMachineType.STANDARD,
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            role=sfn_role,
            timeout=Duration.minutes(30),
            comment="Sunday Album photo processing pipeline",
        )

        # Wire state machine ARN into sa-jobs Lambda
        jobs_fn.add_environment("STATE_MACHINE_ARN", state_machine.state_machine_arn)

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
            self, "ApiUrl",
            value=http_api.api_endpoint,
            export_name="SundayAlbumApiUrl",
            description="HTTP API base URL",
        )
        CfnOutput(
            self, "BucketName",
            value=bucket.bucket_name,
            export_name="SundayAlbumBucketName",
            description="S3 data bucket",
        )
        CfnOutput(
            self, "SessionsTableName",
            value=sessions_table.table_name,
            export_name="SundayAlbumSessionsTable",
        )
        CfnOutput(
            self, "JobsTableName",
            value=jobs_table.table_name,
            export_name="SundayAlbumJobsTable",
        )
        CfnOutput(
            self, "StateMachineArn",
            value=state_machine.state_machine_arn,
            export_name="SundayAlbumStateMachineArn",
            description="Step Functions state machine ARN",
        )
        CfnOutput(
            self, "AuthFunctionName",
            value=auth_fn.function_name,
            export_name="SundayAlbumAuthFunction",
        )
        CfnOutput(
            self, "JobsFunctionName",
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
        self.state_machine = state_machine
