"""计量经济学：OLS 回归与诊断检验。

包含：
- OLS 多元线性回归
- 多重共线性 VIF
- 异方差 Breusch-Pagan 检验
- 自相关 Durbin-Watson 统计量
- 时间序列 ADF 单位根检验
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller


def _prepare_xy(df: pd.DataFrame, y_col: str, x_cols: List[str]):
    """构造回归用的 X、y 矩阵，自动 one-hot 编码并删除缺失行。

    Args:
        df: 输入 DataFrame。
        y_col: 因变量列名。
        x_cols: 自变量列名列表。

    Returns:
        (y, X) 元组，X 已添加常数项。
    """
    data = df[[y_col] + x_cols].dropna()
    y = data[y_col]
    X = data[x_cols]
    # 类别型自变量做 one-hot 编码
    X = pd.get_dummies(X, drop_first=True)
    # 确保为浮点型，避免 statsmodels 对 object/bool 报错
    X = X.astype(float)
    X = sm.add_constant(X)
    return y, X


def ols_regression(df: pd.DataFrame, y_col: str, x_cols: List[str]):
    """执行 OLS 多元线性回归。

    Args:
        df: 输入 DataFrame。
        y_col: 因变量列名。
        x_cols: 自变量列名列表。

    Returns:
        拟合后的 statsmodels 回归结果对象（RegressionResultsWrapper）。

    Raises:
        ValueError: 当有效样本不足时抛出。
    """
    y, X = _prepare_xy(df, y_col, x_cols)
    if len(y) <= X.shape[1]:
        raise ValueError("有效样本量不足以进行回归，请检查缺失值或减少自变量。")
    return sm.OLS(y, X).fit()


def coefficients_table(model) -> pd.DataFrame:
    """提取回归系数表（系数、标准误、t 值、p 值、置信区间）。

    Args:
        model: 已拟合的回归结果对象。

    Returns:
        系数明细 DataFrame。
    """
    conf = model.conf_int()
    table = pd.DataFrame({
        "系数": model.params,
        "标准误": model.bse,
        "t值": model.tvalues,
        "p值": model.pvalues,
        "95%下限": conf[0],
        "95%上限": conf[1],
    })
    table["显著性"] = table["p值"].apply(_significance_stars)
    return table.round(4)


def _significance_stars(p: float) -> str:
    """根据 p 值返回显著性标记。"""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def model_metrics(model) -> dict:
    """提取模型整体拟合指标。

    Args:
        model: 已拟合的回归结果对象。

    Returns:
        包含 R²、调整 R²、F 统计量、AIC、BIC 等的字典。
    """
    return {
        "R²": round(model.rsquared, 4),
        "调整R²": round(model.rsquared_adj, 4),
        "F统计量": round(model.fvalue, 4),
        "F检验p值": round(model.f_pvalue, 6),
        "AIC": round(model.aic, 2),
        "BIC": round(model.bic, 2),
        "样本量": int(model.nobs),
    }


def vif_table(df: pd.DataFrame, x_cols: List[str]) -> pd.DataFrame:
    """计算自变量的方差膨胀因子(VIF)，用于诊断多重共线性。

    经验上 VIF > 10 表示存在严重多重共线性。

    Args:
        df: 输入 DataFrame。
        x_cols: 自变量列名列表。

    Returns:
        每个自变量的 VIF 值 DataFrame。
    """
    data = df[x_cols].dropna()
    X = pd.get_dummies(data, drop_first=True).astype(float)
    X = sm.add_constant(X)
    rows = []
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        rows.append({
            "变量": name,
            "VIF": round(variance_inflation_factor(X.values, i), 4),
        })
    return pd.DataFrame(rows)


def diagnostics(model, df: pd.DataFrame, y_col: str, x_cols: List[str]) -> dict:
    """对回归模型进行诊断检验。

    包含异方差(Breusch-Pagan)与自相关(Durbin-Watson)检验。

    Args:
        model: 已拟合的回归结果对象。
        df: 输入 DataFrame。
        y_col: 因变量列名。
        x_cols: 自变量列名列表。

    Returns:
        诊断结果字典。
    """
    _, X = _prepare_xy(df, y_col, x_cols)
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X)
    dw = durbin_watson(model.resid)
    return {
        "Breusch-Pagan统计量": round(float(bp_stat), 4),
        "BP检验p值": round(float(bp_p), 4),
        "异方差结论": "存在异方差" if bp_p < 0.05 else "未发现异方差",
        "Durbin-Watson": round(float(dw), 4),
        "自相关结论": _dw_conclusion(dw),
    }


def _dw_conclusion(dw: float) -> str:
    """根据 Durbin-Watson 值给出自相关结论（经验判断）。"""
    if dw < 1.5:
        return "可能存在正自相关"
    if dw > 2.5:
        return "可能存在负自相关"
    return "无明显自相关"


def adf_test(series: pd.Series) -> dict:
    """对时间序列进行 ADF 单位根检验（平稳性检验）。

    Args:
        series: 待检验的数值序列。

    Returns:
        ADF 统计量、p 值与平稳性结论的字典。
    """
    s = series.dropna()
    stat, p, lags, nobs, crit, _ = adfuller(s, autolag="AIC")
    return {
        "ADF统计量": round(float(stat), 4),
        "p值": round(float(p), 4),
        "滞后阶数": int(lags),
        "样本量": int(nobs),
        "1%临界值": round(crit["1%"], 4),
        "5%临界值": round(crit["5%"], 4),
        "结论": "平稳" if p < 0.05 else "非平稳（存在单位根）",
    }
