"""AWS CDK stack for the Sales Deal Acceleration Agent.

Provisions:
- S3 bucket for document storage (synced from data/documents/)
- ECR repository for the Docker image
- ECS Fargate service running the Streamlit app
- Application Load Balancer for public access
- Secrets Manager for API keys
"""

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class SalesDealAgentStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # --- S3: Document Storage ---
        docs_bucket = s3.Bucket(
            self,
            "DocsBucket",
            bucket_name=f"sales-deal-agent-docs-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        s3deploy.BucketDeployment(
            self,
            "DeployDocs",
            sources=[s3deploy.Source.asset("../data/documents")],
            destination_bucket=docs_bucket,
            destination_key_prefix="documents",
        )

        # --- Secrets Manager: API Keys ---
        api_secrets = secretsmanager.Secret(
            self,
            "ApiSecrets",
            secret_name="sales-deal-agent/api-keys",
            description="API keys for the Sales Deal Acceleration Agent",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"AWS_ACCESS_KEY_ID":"","AWS_SECRET_ACCESS_KEY":"","TAVILY_API_KEY":"","LANGCHAIN_API_KEY":"","OPENAI_API_KEY":"","ANTHROPIC_API_KEY":""}',
                generate_string_key="_placeholder",
            ),
        )

        # --- VPC ---
        vpc = ec2.Vpc(
            self,
            "AgentVpc",
            max_azs=2,
            nat_gateways=1,
        )

        # --- ECS Cluster ---
        cluster = ecs.Cluster(
            self,
            "AgentCluster",
            vpc=vpc,
            cluster_name="sales-deal-agent",
        )

        # --- Docker Image (built from project root) ---
        image = ecr_assets.DockerImageAsset(
            self,
            "AgentImage",
            directory="..",
            file="Dockerfile",
        )

        # --- ECS Fargate Service + ALB ---
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "AgentService",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(image),
                container_port=8501,
                environment={
                    "LLM_PROVIDER": "bedrock",
                    "AWS_DEFAULT_REGION": "us-east-1",
                    "DOCUMENT_SOURCE": "s3",
                    "S3_BUCKET_NAME": docs_bucket.bucket_name,
                    "SEARCH_PROVIDER": "tavily",
                    "LANGCHAIN_TRACING_V2": "true",
                    "LANGCHAIN_PROJECT": "sales-deal-acceleration-agent",
                },
                secrets={
                    "TAVILY_API_KEY": ecs.Secret.from_secrets_manager(api_secrets, "TAVILY_API_KEY"),
                    "LANGCHAIN_API_KEY": ecs.Secret.from_secrets_manager(api_secrets, "LANGCHAIN_API_KEY"),
                },
            ),
            public_load_balancer=True,
        )

        # Health check configuration
        fargate_service.target_group.configure_health_check(
            path="/_stcore/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(5),
        )

        # Grant S3 read access to the Fargate task
        docs_bucket.grant_read(fargate_service.task_definition.task_role)

        # Grant Bedrock invoke access for LLM and embeddings
        fargate_service.task_definition.task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

        # --- Outputs ---
        CfnOutput(self, "LoadBalancerDNS",
                  value=fargate_service.load_balancer.load_balancer_dns_name,
                  description="Application URL")
        CfnOutput(self, "S3BucketName",
                  value=docs_bucket.bucket_name,
                  description="Document storage bucket")
        CfnOutput(self, "SecretsArn",
                  value=api_secrets.secret_arn,
                  description="Secrets Manager ARN — update with real API keys before deploying")
