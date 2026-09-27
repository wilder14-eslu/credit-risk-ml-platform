import os


class Settings:
    app_name = os.getenv("APP_NAME", "Credit Risk ML Platform")
    model_path = os.getenv("MODEL_PATH", "data/processed/model.joblib")



settings = Settings()
