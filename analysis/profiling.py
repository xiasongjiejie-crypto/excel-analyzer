"""数据画像与质量检查。

提供数据集的总体概览：维度、缺失情况、重复行、异常值等。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def overview(df: pd.DataFrame) -> dict:
    """返回数据集的整体概览信息。

    Args:
        df: 输入 DataFrame。

    Returns:
        包含行数、列数、内存占用、重复行数的字典。
    """
    return {
        "行数": int(df.shape[0]),
        "列数": int(df.shape[1]),
        "重复行数": int(df.duplicated().sum()),
        "内存占用(KB)": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """生成逐列的画像表。

    包含：数据类型、非空数量、缺失数量、缺失率、唯一值数量。

    Args:
        df: 输入 DataFrame。

    Returns:
        逐列汇总的 DataFrame。
    """
    total = len(df)
    summary = pd.DataFrame({
        "数据类型": df.dtypes.astype(str),
        "非空数量": df.notna().sum(),
        "缺失数量": df.isnull().sum(),
        "缺失率(%)": (df.isnull().sum() / total * 100).round(2) if total else 0,
        "唯一值数量": df.nunique(),
    })
    return summary.reset_index().rename(columns={"index": "列名"})


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """使用 IQR 方法检测数值列的异常值数量。

    异常值定义为：小于 Q1 - 1.5*IQR 或 大于 Q3 + 1.5*IQR。

    Args:
        df: 输入 DataFrame。

    Returns:
        每个数值列的异常值数量与占比的 DataFrame。
    """
    num_cols = df.select_dtypes(include="number").columns
    rows = []
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((series < lower) | (series > upper)).sum())
        rows.append({
            "列名": col,
            "异常值数量": n_out,
            "异常值占比(%)": round(n_out / len(series) * 100, 2),
            "下界": round(lower, 4),
            "上界": round(upper, 4),
        })
    return pd.DataFrame(rows)
