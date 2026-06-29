"""扩展模型库。

提供更多统计与机器学习模型：
- 二元离散选择：Logit / Probit
- 时间序列预测：ARIMA
- 无监督学习：KMeans 聚类、PCA 主成分分析
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 二元离散选择模型
# ---------------------------------------------------------------------------
def _prepare_xy(df: pd.DataFrame, y_col: str, x_cols: List[str]):
    """构造离散选择模型的 X、y。

    Args:
        df: 输入 DataFrame。
        y_col: 二元因变量列名（取值需为 0/1）。
        x_cols: 自变量列名列表。

    Returns:
        (y, X) 元组，X 已 one-hot 编码并添加常数项。
    """
    data = df[[y_col] + x_cols].dropna()
    y = data[y_col]
    X = pd.get_dummies(data[x_cols], drop_first=True).astype(float)
    X = sm.add_constant(X)
    return y, X


def discrete_choice(df: pd.DataFrame, y_col: str, x_cols: List[str],
                    model: str = "logit"):
    """拟合二元离散选择模型（Logit 或 Probit）。

    Args:
        df: 输入 DataFrame。
        y_col: 二元因变量列名（取值应为 0/1）。
        x_cols: 自变量列名列表。
        model: "logit" 或 "probit"。

    Returns:
        拟合后的结果对象。

    Raises:
        ValueError: 当因变量非二元或模型名非法时抛出。
    """
    y, X = _prepare_xy(df, y_col, x_cols)
    unique = set(pd.unique(y.dropna()))
    if not unique.issubset({0, 1}):
        raise ValueError("因变量必须为二元变量（取值仅 0 和 1）。")

    if model == "logit":
        return sm.Logit(y, X).fit(disp=False)
    if model == "probit":
        return sm.Probit(y, X).fit(disp=False)
    raise ValueError(f"不支持的模型类型：{model}")


def discrete_choice_summary(result) -> pd.DataFrame:
    """提取离散选择模型的系数表与边际意义。

    Args:
        result: 已拟合的 Logit/Probit 结果对象。

    Returns:
        系数、标准误、z 值、p 值与几率比(odds ratio)的 DataFrame。
    """
    table = pd.DataFrame({
        "系数": result.params,
        "标准误": result.bse,
        "z值": result.tvalues,
        "p值": result.pvalues,
        "几率比(OR)": np.exp(result.params),
    })
    return table.round(4)


# ---------------------------------------------------------------------------
# 时间序列：ARIMA
# ---------------------------------------------------------------------------
def arima_forecast(series: pd.Series, order=(1, 1, 1), steps: int = 10):
    """拟合 ARIMA 模型并预测未来若干期。

    Args:
        series: 时间序列（数值）。
        order: ARIMA 的 (p, d, q) 阶数。
        steps: 向前预测的期数。

    Returns:
        (拟合结果对象, 预测 DataFrame) 元组；预测含均值与置信区间。
    """
    from statsmodels.tsa.arima.model import ARIMA

    s = series.dropna().reset_index(drop=True)
    fitted = ARIMA(s, order=order).fit()
    fc = fitted.get_forecast(steps=steps)
    pred = pd.DataFrame({
        "预测值": fc.predicted_mean,
        "下限95%": fc.conf_int().iloc[:, 0],
        "上限95%": fc.conf_int().iloc[:, 1],
    })
    pred.index = range(len(s), len(s) + steps)
    return fitted, pred.round(4)


# ---------------------------------------------------------------------------
# 无监督学习：KMeans 聚类
# ---------------------------------------------------------------------------
def kmeans_cluster(df: pd.DataFrame, feature_cols: List[str], n_clusters: int = 3):
    """对指定特征执行 KMeans 聚类（自动标准化）。

    Args:
        df: 输入 DataFrame。
        feature_cols: 参与聚类的数值特征列。
        n_clusters: 聚类簇数。

    Returns:
        (带 cluster 列的 DataFrame, 各簇样本数 Series, 惯性 inertia) 元组。
    """
    data = df[feature_cols].dropna()
    scaled = StandardScaler().fit_transform(data)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)

    result = data.copy()
    result["cluster"] = labels
    counts = pd.Series(labels).value_counts().sort_index()
    counts.index = [f"簇 {i}" for i in counts.index]
    return result, counts, round(float(km.inertia_), 4)


# ---------------------------------------------------------------------------
# 降维：PCA 主成分分析
# ---------------------------------------------------------------------------
def pca_analysis(df: pd.DataFrame, feature_cols: List[str], n_components: int = 2):
    """对指定特征执行 PCA 主成分分析（自动标准化）。

    Args:
        df: 输入 DataFrame。
        feature_cols: 参与降维的数值特征列。
        n_components: 主成分数量。

    Returns:
        (主成分得分 DataFrame, 方差解释率表 DataFrame) 元组。
    """
    data = df[feature_cols].dropna()
    scaled = StandardScaler().fit_transform(data)
    n_components = min(n_components, len(feature_cols))
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    score_df = pd.DataFrame(
        scores, columns=[f"PC{i + 1}" for i in range(n_components)]
    )
    var_df = pd.DataFrame({
        "主成分": [f"PC{i + 1}" for i in range(n_components)],
        "方差解释率(%)": (pca.explained_variance_ratio_ * 100).round(2),
        "累计解释率(%)": (np.cumsum(pca.explained_variance_ratio_) * 100).round(2),
    })
    return score_df, var_df
