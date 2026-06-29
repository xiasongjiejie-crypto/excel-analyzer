"""面板数据（Panel Data）分析。

面板数据同时包含「个体（横截面）」与「时间」两个维度。本模块基于
linearmodels 库提供：
- 混合 OLS（Pooled OLS）
- 固定效应模型（Fixed Effects, FE）—— 控制个体不随时间变化的异质性
- 随机效应模型（Random Effects, RE）
- Hausman 检验 —— 帮助在 FE 与 RE 之间做选择

数据要求：需指定「个体标识列」与「时间标识列」，二者共同构成面板索引。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def prepare_panel(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_col: str,
    x_cols: List[str],
) -> Tuple[pd.Series, pd.DataFrame]:
    """构造面板数据所需的 (y, X)，并设置 (个体, 时间) 双重索引。

    Args:
        df: 原始 DataFrame。
        entity_col: 个体标识列名（如公司、省份、个人 ID）。
        time_col: 时间标识列名（如年份、季度）。
        y_col: 因变量列名。
        x_cols: 自变量列名列表。

    Returns:
        (y, X) 元组，索引为 MultiIndex(entity, time)，X 已 one-hot 编码。

    Raises:
        ValueError: 当有效样本不足或时间维度不足时抛出。
    """
    cols = [entity_col, time_col, y_col] + x_cols
    data = df[cols].dropna().copy()
    if data.empty:
        raise ValueError("去除缺失值后无有效数据，请检查所选列。")

    # 时间列尝试转为数值或日期，确保可排序
    data[time_col] = _coerce_time(data[time_col])
    data = data.set_index([entity_col, time_col])

    y = data[y_col]
    X = data[x_cols]
    X = pd.get_dummies(X, drop_first=True).astype(float)

    n_entity = data.index.get_level_values(0).nunique()
    n_time = data.index.get_level_values(1).nunique()
    if n_entity < 2 or n_time < 2:
        raise ValueError(
            f"面板维度不足（个体数={n_entity}, 时间数={n_time}），"
            "面板模型要求个体与时间均至少为 2。"
        )
    return y, X


def _coerce_time(series: pd.Series) -> pd.Series:
    """将时间列转换为可排序类型（数值优先，其次日期）。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() > 0.8:
        return dt
    return series  # 退回原值（作为分类索引）


def panel_dimensions(df, entity_col: str, time_col: str) -> dict:
    """返回面板数据的维度信息。

    Args:
        df: 原始 DataFrame。
        entity_col: 个体标识列名。
        time_col: 时间标识列名。

    Returns:
        含个体数、时间期数、是否平衡面板的字典。
    """
    n_entity = df[entity_col].nunique()
    n_time = df[time_col].nunique()
    # 平衡面板：每个个体的观测数都等于时间期数
    counts = df.groupby(entity_col)[time_col].nunique()
    balanced = bool((counts == n_time).all())
    return {
        "个体数": int(n_entity),
        "时间期数": int(n_time),
        "总观测数": int(len(df)),
        "面板类型": "平衡面板" if balanced else "非平衡面板",
    }


# ---------------------------------------------------------------------------
# 三类面板模型
# ---------------------------------------------------------------------------
def pooled_ols(y: pd.Series, X: pd.DataFrame):
    """混合 OLS（Pooled OLS）：忽略个体差异，直接合并回归。"""
    from linearmodels.panel import PooledOLS
    from linearmodels.panel import PanelOLS  # noqa: F401  确保依赖可用

    Xc = _add_const(X)
    return PooledOLS(y, Xc).fit()


def fixed_effects(y: pd.Series, X: pd.DataFrame):
    """固定效应模型（FE）：控制个体不随时间变化的固定异质性。"""
    from linearmodels.panel import PanelOLS

    Xc = _add_const(X)
    return PanelOLS(y, Xc, entity_effects=True, drop_absorbed=True).fit()


def random_effects(y: pd.Series, X: pd.DataFrame):
    """随机效应模型（RE）：假设个体效应与自变量不相关。"""
    from linearmodels.panel import RandomEffects

    Xc = _add_const(X)
    return RandomEffects(y, Xc).fit()


def _add_const(X: pd.DataFrame) -> pd.DataFrame:
    """为自变量矩阵添加常数项列。"""
    Xc = X.copy()
    Xc.insert(0, "const", 1.0)
    return Xc


# ---------------------------------------------------------------------------
# 结果提取与检验
# ---------------------------------------------------------------------------
def coef_table(result) -> pd.DataFrame:
    """提取面板模型的系数表。

    Args:
        result: linearmodels 拟合结果对象。

    Returns:
        含系数、标准误、t 值、p 值与显著性标记的 DataFrame。
    """
    table = pd.DataFrame({
        "系数": result.params,
        "标准误": result.std_errors,
        "t值": result.tstats,
        "p值": result.pvalues,
    })
    table["显著性"] = table["p值"].apply(_stars)
    return table.round(4)


def _stars(p: float) -> str:
    """根据 p 值返回显著性标记。"""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def model_metrics(result) -> dict:
    """提取面板模型的整体拟合指标。

    Args:
        result: linearmodels 拟合结果对象。

    Returns:
        含 R²、观测数、F 统计量等的字典。
    """
    metrics = {
        "R²": round(float(result.rsquared), 4),
        "观测数": int(result.nobs),
    }
    # 组内 R²（FE 模型常用）
    if hasattr(result, "rsquared_within") and result.rsquared_within is not None:
        metrics["组内R²"] = round(float(result.rsquared_within), 4)
    if hasattr(result, "rsquared_between") and result.rsquared_between is not None:
        metrics["组间R²"] = round(float(result.rsquared_between), 4)
    # F 检验
    try:
        metrics["F统计量"] = round(float(result.f_statistic.stat), 4)
        metrics["F检验p值"] = round(float(result.f_statistic.pval), 6)
    except Exception:  # noqa: BLE001
        pass
    return metrics


def hausman_test(fe_result, re_result) -> dict:
    """Hausman 检验：在固定效应与随机效应之间做选择。

    原假设 H0：随机效应模型一致且有效（即应选 RE）。
    若 p < 0.05 拒绝 H0，应选择固定效应模型（FE）。

    Args:
        fe_result: 固定效应模型结果。
        re_result: 随机效应模型结果。

    Returns:
        含统计量、自由度、p 值与结论的字典。
    """
    # 取两模型共有且非常数项的系数
    common = [
        c for c in fe_result.params.index
        if c in re_result.params.index and c != "const"
    ]
    if not common:
        return {"结论": "无可比较的共同系数，无法进行 Hausman 检验"}

    b_fe = fe_result.params[common].values
    b_re = re_result.params[common].values
    cov_fe = fe_result.cov.loc[common, common].values
    cov_re = re_result.cov.loc[common, common].values

    diff = b_fe - b_re
    cov_diff = cov_fe - cov_re
    # 用伪逆增强数值稳定性（协方差差可能近奇异）
    stat = float(diff @ np.linalg.pinv(cov_diff) @ diff)
    dof = len(common)
    p = float(stats.chi2.sf(stat, dof))

    return {
        "Hausman统计量": round(stat, 4),
        "自由度": dof,
        "p值": round(p, 4),
        "结论": "应选择固定效应(FE)" if p < 0.05 else "应选择随机效应(RE)",
    }
