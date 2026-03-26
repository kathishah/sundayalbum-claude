#!/usr/bin/env python3
import aws_cdk as cdk

from infra.sundayalbum_stack import SundayAlbumStack

app = cdk.App()
SundayAlbumStack(
    app,
    "SundayAlbumStack",
    env=cdk.Environment(account="680073251743", region="us-west-2"),
)
app.synth()
