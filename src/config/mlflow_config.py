"""
MLflow Configuration Module

This module configures MLflow to use PostgreSQL as backend store
and MinIO as artifact store based on the docker-compose setup.
"""

import os

import mlflow


def setup_mlflow():
    """
    Configure MLflow tracking URI and artifact storage.

    Uses environment variables with sensible defaults for the docker-compose setup.
    """
    # PostgreSQL backend store configuration
    postgres_user = os.getenv("MLFLOW_POSTGRES_USER", "mlflow")
    postgres_password = os.getenv("MLFLOW_POSTGRES_PASSWORD", "mlflow")
    postgres_host = os.getenv("MLFLOW_POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("MLFLOW_POSTGRES_PORT", "5432")
    postgres_db = os.getenv("MLFLOW_POSTGRES_DB", "mlflow_db")

    # Construct PostgreSQL connection string
    tracking_uri = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # MinIO (S3) artifact storage configuration
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
    )
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
        "AWS_SECRET_ACCESS_KEY", "minioadmin"
    )

    print("✓ MLflow configured:")
    print(f"  - Tracking URI: {tracking_uri}")
    print(f"  - S3 Endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
    print(f"  - Artifact Location: s3://mlflow-artifacts/")


def get_artifact_location():
    """Return the S3 artifact location for MLflow experiments."""
    return "s3://mlflow-artifacts/"
