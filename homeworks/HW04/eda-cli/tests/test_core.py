from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():

    # НАЧАЛО НОВОЕ
    df = _sample_df()
    missing_df = missing_table(df)
    summary = summarize_dataset(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1
    
    flags = compute_quality_flags(df,summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0

    # Проверяем, что новые ключи появились в словаре (даже если они False)
    assert "has_constant_columns" in flags
    assert "has_full_duplicates" in flags
    # КОНЕЦ НОВОЕ


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# НОВЫЙ ТЕСТ (2.3.3)
def test_new_heuristics_logic():
    """
    Проверяем, что новые эвристики (константы и дубликаты) реально работают
    на специально подготовленных данных.
    """
    # Создаем DataFrame с проблемами:
    # Дубликат: строки 0 и 1 полностью совпадают.
    # Константа: колонка 'const_col' имеет везде значение 99.
    df_bad = pd.DataFrame({
        "col_a": [1, 1, 2, 3],
        "col_b": ["x", "x", "y", "z"],
        "const_col": [99, 99, 99, 99]
    })

    summary = summarize_dataset(df_bad)
    missing_df = missing_table(df_bad)
    
    flags = compute_quality_flags(df_bad, summary, missing_df)

    # Проверка на константную колонку
    assert flags["has_constant_columns"] is True
    assert flags["constant_columns_count"] == 1

    # Проверка на дубликаты
    assert flags["has_full_duplicates"] is True
    assert flags["duplicates_count"] > 0

    # Проверка скора
    assert flags["quality_score"] < 1.0

    # Проверка на "чистом" датасете (чтобы убедиться, что флаги не горят всегда)
    df_good = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    flags_good = compute_quality_flags(df_good, summarize_dataset(df_good), missing_table(df_good))
    
    assert flags_good["has_constant_columns"] is False
    assert flags_good["has_full_duplicates"] is False