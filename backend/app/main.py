import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.api import get_active_model, router, train_and_store
from app.services.weather import synthetic_weather_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not get_active_model():
        try:
            train_and_store(synthetic_weather_data())
        except Exception:
            logging.getLogger(__name__).exception("Initial sample model training failed")
    yield


app = FastAPI(
    title="Localized Climate & Weather Forecast Engine",
    description="Multi-output weather regression with versioned training and model tracking.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(router)


@app.get("/")
def root():
    return {"name": "Localized Climate & Weather Forecast Engine", "docs": "/docs", "health": "/api/health"}
