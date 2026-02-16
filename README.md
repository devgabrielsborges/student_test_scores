# Student Test Scores Prediction

Machine learning project for predicting student exam scores using various regression models with hyperparameter optimization.

## Overview

This project implements multiple regression models to predict student exam scores based on various features. All models use:
- **Optuna** for hyperparameter optimization
- **MLflow** for experiment tracking (PostgreSQL + MinIO backend)
- **Docker Compose** for infrastructure management
- Cross-validation with RMSE optimization

## Project Structure

```
student_test_scores/
├── compose.yml              # Docker Compose configuration
├── Makefile                 # Convenient commands for common tasks
├── MLFLOW_SETUP.md         # Detailed MLflow setup documentation
├── data/
│   ├── raw/                # Raw competition data
│   └── processed/          # Preprocessed data
├── src/
│   ├── config/
│   │   └── mlflow_config.py    # MLflow PostgreSQL + MinIO configuration
│   ├── models/             # Model training scripts
│   │   ├── linear.py      # Linear Regression (4 trials)
│   │   ├── ridge.py       # Ridge Regression (200 trials)
│   │   ├── sgdc.py        # SGD Regressor (200 trials)
│   │   ├── svm.py         # Support Vector Regression (150 trials)
│   │   ├── rf.py          # Random Forest (300 trials)
│   │   ├── gb.py          # Gradient Boosting (300 trials)
│   │   └── xgb.py         # XGBoost (300 trials)
│   ├── preprocessing/
│   │   └── preprocess.py  # Data preprocessing pipeline
│   └── notebooks/
│       └── eda.ipynb      # Exploratory data analysis
└── pyproject.toml         # Python dependencies
```

## Quick Start

### 1. Prerequisites

- Python 3.13+
- Docker and Docker Compose
- `uv` package manager (or pip)

### 2. Setup Infrastructure

```bash
# Start PostgreSQL and MinIO
make start

# Or manually:
docker compose up -d
```

### 3. Install Dependencies

```bash
make install

# Or manually:
uv sync
```

### 4. Train Models

```bash
# Train individual models
cd src/models
python linear.py
python xgb.py
python rf.py

# Or train all models
make train-all
```

### 5. View Results

```bash
# Start MLflow UI
make ui

# Open http://localhost:5000 in your browser
```

## Models

All models are optimized using Optuna with cross-validation on RMSE:

| Model | Trials | Key Hyperparameters |
|-------|--------|-------------------|
| **Linear Regression** | 4 | fit_intercept, positive |
| **Ridge Regression** | 200 | alpha, solver, max_iter |
| **SGD Regressor** | 200 | loss, penalty, alpha, learning_rate |
| **SVM** | 150 | C, kernel, gamma, epsilon |
| **Random Forest** | 300 | n_estimators, max_depth, min_samples_split |
| **Gradient Boosting** | 300 | n_estimators, learning_rate, max_depth |
| **XGBoost** | 300 | n_estimators, learning_rate, max_depth, subsample |

## Data Preprocessing

The preprocessing pipeline handles:
- Ordinal feature encoding (sleep_quality, facility_rating, exam_difficulty)
- One-hot encoding for nominal features (gender, course, study_method)
- Standard scaling of all features
- Train/test splitting with stratification

## Submission Files

Each model automatically generates a Kaggle-compatible submission file:
- Format: `id,exam_score`
- Stored locally and logged to MLflow as artifacts
- Ready for direct upload to Kaggle

## MLflow Configuration

This project uses a production-ready MLflow setup:

- **Backend Store**: PostgreSQL (experiment metadata, parameters, metrics)
- **Artifact Store**: MinIO S3-compatible storage (models, plots, submission files)
- **Tracking**: Automatic logging of all experiments to the configured backend

See [MLFLOW_SETUP.md](MLFLOW_SETUP.md) for detailed configuration.

## Available Commands

```bash
make help       # Show all available commands
make start      # Start infrastructure (PostgreSQL + MinIO)
make stop       # Stop infrastructure
make status     # Check service health
make ui         # Launch MLflow UI
make clean      # Remove all data (destructive!)
make install    # Install Python dependencies
make train-all  # Train all models sequentially
```

## Development Workflow

1. **Start infrastructure**: `make start`
2. **Explore data**: Open `src/notebooks/eda.ipynb`
3. **Train models**: Run training scripts in `src/models/`
4. **Compare results**: Use MLflow UI (`make ui`)
5. **Submit predictions**: Use generated CSV files in `src/models/`

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Monitoring

### Check Service Status
```bash
make status
```

### View Logs
```bash
make logs
```

### Access MinIO Console
- URL: http://localhost:9001
- Username: `minioadmin`
- Password: `minioadmin`

## Troubleshooting

### Services won't start
```bash
# Check Docker is running
docker compose ps

# View logs
docker compose logs
```

### Database connection errors
```bash
# Restart PostgreSQL
docker compose restart postgres

# Check health
docker exec mlflow-postgres pg_isready -U mlflow
```

### MinIO artifacts not saving
```bash
# Check MinIO bucket exists
docker compose logs minio-setup

# Verify credentials match in configuration
```

## Contributing

1. Ensure all services are running (`make start`)
2. Install dependencies (`make install`)
3. Make your changes
4. Test with a training run
5. Verify in MLflow UI

## License

MIT

## Acknowledgments

- Kaggle Playground Series S6E1 for the dataset
- MLflow for experiment tracking
- Optuna for hyperparameter optimization
