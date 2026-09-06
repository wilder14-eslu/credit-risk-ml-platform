import os


class Settings:
    app_name = os.getenv("APP_NAME", "Credit Risk ML Platform")
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://credit_risk:credit_risk@localhost:5432/credit_risk",
    )
    model_path = os.getenv("MODEL_PATH", "data/processed/model.joblib")



settings = Settings()
