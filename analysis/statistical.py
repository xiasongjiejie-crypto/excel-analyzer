"""假设检验与相关性分析。

提供正态性检验、相关性矩阵计算等统计推断功能。
"""
from __future__ import annotations

import pandas as pd
from scipy import stats


def normality_test(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """对各数值列进行正态性检验。

    样本量 <= 5000 时使用 Shapiro-Wilk 检验，否则使用
    D'Agostino-Pearson (normaltest) 检验。

    Args:
        df: 输入 DataFrame。
        alpha: 显著性水平，默认 0.05。

    Returns:
        每个数值列的检验方法、统计量、p 值与是否服从正态的 DataFrame。
    """
    num_cols = df.select_dtypes(include="number").columns
    rows = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        if len(series) <= 5000:
            method, (stat, p) = "Shapiro-Wilk", stats.shapiro(series)
        else:
            method, (stat, p) = "D'Agostino", stats.normaltest(series)
        rows.append({
            "列名": col,
            "检验方法": method,
            "统计量": round(float(stat), 4),
            "p值": round(float(p), 4),
            "是否正态": "是" if p > alpha else "否",
        })
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """计算数值列之间的相关系数矩阵。

    Args:
        df: 输入 DataFrame。
        method: 相关系数类型，可选 "pearson" / "spearman" / "kendall"。

    Returns:
        相关系数矩阵 DataFrame。
    """
    num = df.select_dtypes(include="number")
    return num.corr(method=method).round(4)


def top_correlations(df: pd.DataFrame, method: str = "pearson", top_n: int = 10) -> pd.DataFrame:
    """提取相关性最强的若干变量对。

    Args:
        df: 输入 DataFrame。
        method: 相关系数类型。
        top_n: 返回的变量对数量。

    Returns:
        按相关系数绝对值降序排列的变量对 DataFrame。
    """
    corr = correlation_matrix(df, method=method)
    if corr.empty:
        return pd.DataFrame()

    pairs = corr.unstack()  # 转为 (var1, var2) -> 系数 的 Series
    # 去掉自相关与重复对
    pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
    result = (
        pairs.reindex(pairs.abs().sort_values(ascending=False).index)
        .head(top_n)
        .reset_index()
    )
    result.columns = ["变量1", "变量2", "相关系数"]
    return result
