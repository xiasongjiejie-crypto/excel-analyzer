"""Excel 读取与数据清洗工具。

负责：
1. 读取上传的 Excel 文件（支持多 Sheet、.xlsx / .xls）
2. 自动类型推断（日期、数值）
3. 缺失值处理
"""
from __future__ import annotations

from typing import List

import pandas as pd


def list_sheet_names(file) -> List[str]:
    """返回 Excel 文件中的所有工作表名称。

    Args:
        file: 文件路径或类文件对象（如 Streamlit 的 UploadedFile）。

    Returns:
        工作表名称列表。
    """
    return pd.ExcelFile(file).sheet_names


def read_excel(file, sheet_name=0) -> pd.DataFrame:
    """读取指定工作表为 DataFrame。

    Args:
        file: 文件路径或类文件对象。
        sheet_name: 工作表名称或索引，默认第一个。

    Returns:
        读取后的 DataFrame。
    """
    df = pd.read_excel(file, sheet_name=sheet_name)
    # 去除完全空白的列（常见于 Excel 多余空列）
    df = df.dropna(axis=1, how="all")
    return df


def infer_and_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """自动推断并转换列类型。

    - 尝试把 object 列转换为日期；
    - 尝试把伪数值（含文本的数字）转换为数值。

    Args:
        df: 原始 DataFrame。

    Returns:
        类型转换后的新 DataFrame（不修改原对象）。
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != "object":
            continue
        # 尝试转日期
        converted = pd.to_datetime(out[col], errors="coerce")
        if converted.notna().mean() > 0.8:  # 80% 以上可解析则视为日期列
            out[col] = converted
            continue
        # 尝试转数值
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().mean() > 0.8:
            out[col] = numeric
    return out


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """处理缺失值。

    Args:
        df: 输入 DataFrame。
        strategy: 处理策略，可选 "drop"（删除含缺失的行）、
            "mean"（数值列均值填充）、"median"（数值列中位数填充）。

    Returns:
        处理后的新 DataFrame。

    Raises:
        ValueError: 当 strategy 不在支持范围内时抛出。
    """
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns

    if strategy == "drop":
        return out.dropna()
    if strategy == "mean":
        out[num_cols] = out[num_cols].fillna(out[num_cols].mean())
        return out
    if strategy == "median":
        out[num_cols] = out[num_cols].fillna(out[num_cols].median())
        return out

    raise ValueError(f"不支持的缺失值处理策略：{strategy}")


def split_column_types(df: pd.DataFrame):
    """区分数值列与类别列。

    Args:
        df: 输入 DataFrame。

    Returns:
        (数值列名列表, 类别列名列表) 的元组。
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols
