from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from eda_cli.core import (                                              # импорт функций из ядра HW03
    compute_quality_flags,
    missing_table,
    summarize_dataset,
)

app = FastAPI(title="EDA-CLI Service", description="HTTP-сервис качества данных")

class QualityRequest(BaseModel):                                        #запрос
    n_rows: int
    n_cols: int
    n_missing: int = 0

class QualityResponse(BaseModel):                                        #ответ
    ok_for_model: bool
    quality_score: float
    latency_ms: float
    flags: Dict[str, Any]


def calculate_latency(start_time: float) -> float:
    return round((time.time() - start_time) * 1000, 2)


@app.get("/health")
def health_check():
    """Проверка, что сервис работает"""
    return {"status": "ok", "service": "eda-cli", "version": "0.1.0"}


@app.post("/quality", response_model=QualityResponse)
def predict_quality(request: QualityRequest):
    """
    Расчет качества на основе переданных чисел (без самого файла).
    """
    start_ts = time.time()
    
    score = 1.0
    flags = {}
    
    if request.n_rows < 50:
        score -= 0.5
        flags["too_few_rows"] = True
    else:
        flags["too_few_rows"] = False


    total_cells = request.n_rows * request.n_cols
    missing_share = 0.0
    if total_cells > 0:
        missing_share = request.n_missing / total_cells
    
    score -= missing_share
    score = max(0.0, score)
    
    return QualityResponse(
        ok_for_model=(score > 0.5),
        quality_score=round(score, 2),
        latency_ms=calculate_latency(start_ts),
        flags=flags
    )


@app.post("/quality-from-csv")
def quality_from_csv(file: UploadFile = File(...)):
    """
    Принимает CSV-файл, анализирует его через ядро eda-cli
    и возвращает общий скор и основные флаги.
    """
    start_ts = time.time()
    
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {e}")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    score = flags.get("quality_score", 0.0)
    
    return {
        "filename": file.filename,
        "n_rows": summary.n_rows,
        "n_cols": summary.n_cols,
        "quality_score": round(score, 2),
        "ok_for_model": score > 0.6,
        "latency_ms": calculate_latency(start_ts),
    }


@app.post("/quality-flags-from-csv")                                   # 2.3.2. собственный эндпоинт
def get_full_quality_flags(file: UploadFile = File(...)):
    """
    Возвращает ПОЛНЫЙ список флагов качества (включая константы и дубликаты).
    Специфично для HW04.
    """
    start_ts = time.time()
    
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {e}")

    summary = summarize_dataset(df)                                 # Используем ядро
    missing_df = missing_table(df)
    
    all_flags = compute_quality_flags(df, summary, missing_df)      # вычисление новых эвристики
    
    return {
        "filename": file.filename,
        "flags": all_flags, 
        "latency_ms": calculate_latency(start_ts)
    }