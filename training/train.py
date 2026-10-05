"""Treina e exporta um modelo Prophet para o fechamento diário do BTC-USD."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.serialize import model_to_json


DEFAULT_SOURCE_URL = "https://finance.yahoo.com/quote/BTC-USD/history/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina um Prophet para prever o fechamento diário de uma moeda."
    )
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument(
        "--end",
        default=None,
        help="Data final exclusiva (AAAA-MM-DD). O padrão é a data atual.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV local com colunas Date e Close; evita o download do Yahoo Finance.",
    )
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def _standardize_price_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Converte dados do Yahoo ou de CSV para as colunas Date e Close."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()

    lower_to_original = {str(column).lower(): column for column in frame.columns}
    date_column = lower_to_original.get("date") or lower_to_original.get("datetime")
    close_column = lower_to_original.get("close")

    if date_column is None or close_column is None:
        raise ValueError("Os dados precisam conter as colunas Date e Close.")

    clean = frame[[date_column, close_column]].rename(
        columns={date_column: "Date", close_column: "Close"}
    )
    clean["Date"] = pd.to_datetime(clean["Date"], utc=True, errors="coerce").dt.tz_localize(None)
    clean["Close"] = pd.to_numeric(clean["Close"], errors="coerce")
    clean = clean.dropna().drop_duplicates(subset="Date", keep="last")
    clean = clean.loc[clean["Close"] > 0].sort_values("Date").reset_index(drop=True)

    if clean.empty:
        raise ValueError("Nenhuma observação válida foi encontrada.")
    return clean


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    if args.input_csv:
        if not args.input_csv.exists():
            raise FileNotFoundError(f"CSV não encontrado: {args.input_csv}")
        raw = pd.read_csv(args.input_csv)
        acquisition = f"CSV local: {args.input_csv}"
    else:
        raw = yf.download(
            args.ticker,
            start=args.start,
            end=args.end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            multi_level_index=False,
            timeout=30,
        )
        if raw.empty:
            raise RuntimeError("O Yahoo Finance não retornou dados para os parâmetros informados.")
        acquisition = "Download com yfinance"

    source = {
        "provider": "Yahoo Finance",
        "url": DEFAULT_SOURCE_URL,
        "acquisition": acquisition,
    }
    return _standardize_price_data(raw), source


def to_prophet_frame(prices: pd.DataFrame) -> pd.DataFrame:
    # O log reduz o efeito da escala e impede previsões negativas após a inversão.
    return pd.DataFrame({"ds": prices["Date"], "y": np.log(prices["Close"])})


def make_model() -> Prophet:
    return Prophet(
        growth="linear",
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.80,
    )


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    return {
        "mae_usd": round(float(np.mean(np.abs(errors))), 2),
        "rmse_usd": round(float(math.sqrt(np.mean(np.square(errors)))), 2),
        "mape_percent": round(float(np.mean(np.abs(errors / actual)) * 100), 2),
    }


def evaluate(
    prophet_data: pd.DataFrame, test_days: int
) -> tuple[dict[str, object], pd.DataFrame]:
    if test_days < 7:
        raise ValueError("--test-days deve ser no mínimo 7.")
    if len(prophet_data) < test_days + 365:
        raise ValueError("São necessários pelo menos 365 dias de treino além do teste.")

    train = prophet_data.iloc[:-test_days].copy()
    test = prophet_data.iloc[-test_days:].copy()
    model = make_model()
    model.fit(train)

    forecast = model.predict(test[["ds"]])
    actual = np.exp(test["y"].to_numpy())
    predicted = np.exp(forecast["yhat"].to_numpy())
    lower = np.exp(forecast["yhat_lower"].to_numpy())
    upper = np.exp(forecast["yhat_upper"].to_numpy())

    last_train_close = float(np.exp(train["y"].iloc[-1]))
    naive_forecast = np.full_like(actual, last_train_close)
    naive_metrics = compute_metrics(actual, naive_forecast)
    prophet_metrics = compute_metrics(actual, predicted)
    previous_actual = np.concatenate(([last_train_close], actual[:-1]))
    direction_hits = np.sign(actual - previous_actual) == np.sign(predicted - previous_actual)
    prophet_metrics["directional_accuracy_percent"] = round(
        float(np.mean(direction_hits) * 100), 2
    )
    prophet_metrics["interval_coverage_percent"] = round(
        float(np.mean((actual >= lower) & (actual <= upper)) * 100), 2
    )

    predictions = pd.DataFrame(
        {
            "date": test["ds"].dt.strftime("%Y-%m-%d"),
            "actual_close": actual,
            "predicted_close": predicted,
            "predicted_lower": lower,
            "predicted_upper": upper,
            "naive_last_train_close": naive_forecast,
        }
    )
    validation = {
        "strategy": "holdout cronológico",
        "test_observations": test_days,
        "test_start": test["ds"].min().date().isoformat(),
        "test_end": test["ds"].max().date().isoformat(),
        "prophet": prophet_metrics,
        "naive_baseline": naive_metrics,
    }
    return validation, predictions


def save_validation_plot(predictions: pd.DataFrame, destination: Path) -> None:
    dates = pd.to_datetime(predictions["date"])
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.plot(dates, predictions["actual_close"], label="Real", color="#111827")
    axis.plot(dates, predictions["predicted_close"], label="Prophet", color="#2563eb")
    axis.fill_between(
        dates,
        predictions["predicted_lower"],
        predictions["predicted_upper"],
        color="#93c5fd",
        alpha=0.35,
        label="Intervalo de 80%",
    )
    axis.set(title="Validação cronológica — BTC-USD", xlabel="Data", ylabel="USD")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    prices, source = load_data(args)
    data_path = args.data_dir / f"{args.ticker.lower().replace('-', '_')}_daily.csv"
    prices.to_csv(data_path, index=False)
    prophet_data = to_prophet_frame(prices)

    validation, predictions = evaluate(prophet_data, args.test_days)
    predictions.to_csv(args.results_dir / "validation_predictions.csv", index=False)
    save_validation_plot(predictions, args.results_dir / "validation_plot.png")

    # O artefato de produção usa todos os dados; o holdout acima permanece registrado.
    final_model = make_model()
    final_model.fit(prophet_data)
    model_path = args.artifact_dir / "prophet_model.json"
    model_path.write_text(model_to_json(final_model), encoding="utf-8")

    next_date = prophet_data["ds"].max() + pd.Timedelta(days=1)
    next_forecast = final_model.predict(pd.DataFrame({"ds": [next_date]})).iloc[0]
    example_prediction = {
        "date": next_date.date().isoformat(),
        "predicted_close": round(float(np.exp(next_forecast["yhat"])), 2),
        "predicted_lower": round(float(np.exp(next_forecast["yhat_lower"])), 2),
        "predicted_upper": round(float(np.exp(next_forecast["yhat_upper"])), 2),
        "currency": "USD",
    }
    (args.results_dir / "next_day_prediction.json").write_text(
        json.dumps(example_prediction, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": args.ticker,
        "currency": "USD",
        "frequency": "daily",
        "target": "next daily close",
        "target_transform": "natural_log",
        "source": source,
        "training": {
            "observations": len(prophet_data),
            "start": prophet_data["ds"].min().date().isoformat(),
            "end": prophet_data["ds"].max().date().isoformat(),
        },
        "model": {
            "library": "prophet",
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "interval_width": 0.80,
        },
        "validation": validation,
        "example_prediction": example_prediction,
        "limitations": [
            "O modelo usa somente data e preço histórico de fechamento.",
            "Eventos, notícias, volume e variáveis macroeconômicas não são considerados.",
            "A previsão é experimental e não constitui recomendação financeira.",
        ],
    }
    metadata_path = args.artifact_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Dados: {data_path} ({len(prices)} observações)")
    print(f"Modelo: {model_path}")
    print(f"Metadados: {metadata_path}")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
