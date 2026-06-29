"""描述性统计。

提供数值列的扩展描述性统计指标，以及类别列的频数统计。
"""
from __future__ import annotations

import pandas as pd


def numeric_describe(df: pd.DataFrame) -> pd.DataFrame:
    """计算数值列的扩展描述性统计。

    在 pandas 默认 describe 基础上，补充偏度(skew)、峰度(kurtosis)、
    变异系数(CV)等指标。

    Args:
        df: 输入 DataFrame。

    Returns:
        统计指标为行、列名为列的 DataFrame；若无数值列则返回空表。
    """
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()

    desc = num.describe().T  # count/mean/std/min/25%/50%/75%/max
    desc["偏度"] = num.skew()
    desc["峰度"] = num.kurtosis()
    # 变异系数 = 标准差 / 均值（均值为 0 时置为 NaN）
    desc["变异系数"] = (desc["std"] / desc["mean"]).where(desc["mean"] != 0)

    desc = desc.rename(columns={
        "count": "计数", "mean": "均值", "std": "标准差",
        "min": "最小值", "25%": "Q1", "50%": "中位数",
        "75%": "Q3", "max": "最大值",
    })
    return desc.round(4)


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> dict:
    """统计类别列的频数分布。

    Args:
        df: 输入 DataFrame。
        top_n: 每列返回出现频率最高的前 N 个取值。

    Returns:
        以列名为键、Top-N 频数 DataFrame 为值的字典。
    """
    cat = df.select_dtypes(exclude="number")
    result = {}
    for col in cat.columns:
        vc = df[col].value_counts(dropna=False).head(top_n)
        result[col] = vc.rename_axis(col).reset_index(name="频数")
    return result
