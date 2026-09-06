from fastapi import FastAPI

from src.api.routes import router
from src.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(router)
