#!/usr/bin/env python3
"""CDK app entry point for the Sales Deal Acceleration Agent infrastructure."""

import aws_cdk as cdk

from stack import SalesDealAgentStack

app = cdk.App()
SalesDealAgentStack(
    app,
    "SalesDealAgentStack",
    env=cdk.Environment(region="us-east-1"),
)
app.synth()
