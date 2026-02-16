.PHONY: help start stop restart logs clean ui install

help:
	@echo "Available commands:"
	@echo "  make start      - Start PostgreSQL and MinIO services"
	@echo "  make stop       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View logs from all services"
	@echo "  make clean      - Remove all containers and volumes (destructive!)"
	@echo "  make ui         - Start MLflow UI"
	@echo "  make install    - Install Python dependencies"
	@echo "  make status     - Check service status"

start:
	@echo "Starting MLflow infrastructure..."
	docker compose up -d
	@echo "✓ Services started successfully"
	@echo "PostgreSQL: localhost:5432"
	@echo "MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
	@echo "MinIO API: http://localhost:9000"

stop:
	@echo "Stopping services..."
	docker compose stop
	@echo "✓ Services stopped"

restart:
	@echo "Restarting services..."
	docker compose restart
	@echo "✓ Services restarted"

logs:
	docker compose logs -f

status:
	@echo "Checking service status..."
	@docker compose ps
	@echo ""
	@echo "Testing connections..."
	@docker exec -it mlflow-postgres pg_isready -U mlflow -d mlflow_db 2>/dev/null && echo "✓ PostgreSQL is ready" || echo "✗ PostgreSQL is not ready"
	@curl -s http://localhost:9000/minio/health/live >/dev/null 2>&1 && echo "✓ MinIO is ready" || echo "✗ MinIO is not ready"

clean:
	@echo "⚠️  WARNING: This will delete all data including MLflow experiments and artifacts!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	docker compose down -v
	@echo "✓ All containers and volumes removed"

ui:
	@echo "Starting MLflow UI..."
	@echo "Open http://localhost:5000 in your browser"
	AWS_ACCESS_KEY_ID=minioadmin \
	AWS_SECRET_ACCESS_KEY=minioadmin \
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
	uv run mlflow ui --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow_db \
	          --default-artifact-root s3://mlflow-artifacts/ \
	          --host 0.0.0.0

install:
	@echo "Installing dependencies..."
	uv sync
	@echo "✓ Dependencies installed"

train-all:
	@echo "Training all models..."
	@cd src/models && for model in *.py; do \
		echo "Running $$model..."; \
		uv run $$model || exit 1; \
	done
	@echo "✓ All models trained"
