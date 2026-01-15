from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

# Se Prophet estiver disponível no ambiente:
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

app = FastAPI(title="Revenue Forecast Service")

@app.get("/")
def healthcheck():
    return {"status": "ok"}


class Row(BaseModel):
    date: str
    total_revenue: float
    fresh_oil_revenue: Optional[float] = 0
    uco_revenue: Optional[float] = 0

class ForecastRequest(BaseModel):
    horizon_days: int = 15
    train_window_days: int = 180
    rows: List[Row]

@app.post("/forecast")
def forecast(req: ForecastRequest):
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # preenche datas faltantes com 0 (ajuste se preferir NaN)
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full).rename_axis("date").reset_index()
    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce").fillna(0)

    last_date = df["date"].max()

    # janela móvel recente
    start = last_date - pd.Timedelta(days=req.train_window_days)
    train = df[df["date"] >= start].copy()

    if HAS_PROPHET:
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        train_p = train.rename(columns={"date":"ds","total_revenue":"y"})[["ds","y"]]
        m.fit(train_p)

        future = m.make_future_dataframe(periods=req.horizon_days, freq="D", include_history=False)
        fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]

        out = []
        for _, r in fc.iterrows():
            out.append({
                "date": r["ds"].strftime("%Y-%m-%d"),
                "pred_total_revenue": float(r["yhat"]),
                "p80_lower": float(r["yhat_lower"]),
                "p80_upper": float(r["yhat_upper"]),
            })

        return {"model":"prophet", "last_observed_date": last_date.strftime("%Y-%m-%d"), "forecast": out}

    # fallback simples se Prophet não estiver disponível
    # (média por dia da semana nas últimas 12 semanas)
    train["dow"] = train["date"].dt.dayofweek
    cutoff = last_date - pd.Timedelta(days=84)
    recent = train[train["date"] >= cutoff]
    stats = recent.groupby("dow")["total_revenue"].agg(["mean", lambda x: x.quantile(0.10), lambda x: x.quantile(0.90)])
    stats.columns = ["mean","p10","p90"]

    out = []
    for k in range(1, req.horizon_days+1):
        d = last_date + pd.Timedelta(days=k)
        dow = d.dayofweek
        if dow in stats.index:
            pred = stats.loc[dow,"mean"]
            p10 = stats.loc[dow,"p10"]
            p90 = stats.loc[dow,"p90"]
        else:
            pred = p10 = p90 = 0.0
        out.append({
            "date": d.strftime("%Y-%m-%d"),
            "pred_total_revenue": float(pred),
            "p80_lower": float(p10),
            "p80_upper": float(p90),
        })

    return {"model":"dow_baseline", "last_observed_date": last_date.strftime("%Y-%m-%d"), "forecast": out}
