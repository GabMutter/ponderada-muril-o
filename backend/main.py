"""API de inferência do modelo Prophet treinado para BTC-USD."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from prophet.serialize import model_from_json


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/artifacts/prophet_model.json"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "/app/artifacts/metadata.json"))


class PredictionRequest(BaseModel):
    days: int = Field(default=1, ge=1, le=30)


class DailyPrediction(BaseModel):
    date: str
    predicted_close: float
    predicted_lower: float
    predicted_upper: float


class PredictionResponse(BaseModel):
    ticker: str
    currency: str
    generated_at_utc: str
    predictions: list[DailyPrediction]
    disclaimer: str


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Artefato do modelo não encontrado: {MODEL_PATH}")
    if not METADATA_PATH.is_file():
        raise RuntimeError(f"Metadados não encontrados: {METADATA_PATH}")

    model = model_from_json(MODEL_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    if metadata.get("target_transform") != "natural_log":
        raise RuntimeError("Transformação do alvo ausente ou incompatível.")
    if not metadata.get("training", {}).get("end"):
        raise RuntimeError("Data final do treinamento ausente nos metadados.")
    return model, metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.metadata = load_artifacts()
    yield
    app.state.model = None
    app.state.metadata = None


app = FastAPI(
    title="API de previsão BTC-USD",
    description="Inferência experimental com um modelo Prophet treinado.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["monitoramento"])
async def health(request: Request) -> dict[str, Any]:
    metadata = request.app.state.metadata
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "ticker": metadata["ticker"],
        "trained_until": metadata["training"]["end"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["predição"])
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    model = request.app.state.model
    metadata = request.app.state.metadata
    last_training_date = pd.Timestamp(metadata["training"]["end"])
    future_dates = pd.date_range(
        start=last_training_date + pd.Timedelta(days=1),
        periods=payload.days,
        freq="D",
    )

    forecast = model.predict(pd.DataFrame({"ds": future_dates}))
    predictions = [
        DailyPrediction(
            date=row.ds.date().isoformat(),
            predicted_close=round(float(np.exp(row.yhat)), 2),
            predicted_lower=round(float(np.exp(row.yhat_lower)), 2),
            predicted_upper=round(float(np.exp(row.yhat_upper)), 2),
        )
        for row in forecast.itertuples()
    ]

    return PredictionResponse(
        ticker=metadata["ticker"],
        currency=metadata["currency"],
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        predictions=predictions,
        disclaimer="Predição experimental; não constitui recomendação financeira.",
    )
