"""分析报告生成。

将各项分析结果汇总为自包含的 HTML 报告，供用户下载。
不依赖额外第三方库，使用 pandas 的 to_html 渲染表格。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict

import pandas as pd

_CSS = """
<style>
  body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 40px; color: #222; line-height: 1.6; }
  h1 { color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 8px; }
  h2 { color: #333; margin-top: 32px; border-left: 4px solid #1f77b4;
       padding-left: 10px; }
  table { border-collapse: collapse; margin: 12px 0; width: 100%;
          font-size: 13px; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
  th { background: #f2f6fa; color: #1f77b4; }
  td:first-child, th:first-child { text-align: left; }
  tr:nth-child(even) { background: #fafafa; }
  .meta { color: #888; font-size: 13px; }
  .kv { display: inline-block; background: #f2f6fa; border-radius: 6px;
        padding: 8px 14px; margin: 4px 8px 4px 0; }
  .kv b { color: #1f77b4; }
</style>
"""


def _df_to_html(df: pd.DataFrame) -> str:
    """将 DataFrame 转为 HTML 表格，空表给出提示。"""
    if df is None or df.empty:
        return "<p class='meta'>（无数据）</p>"
    return df.to_html(index=True, border=0, justify="center")


def _dict_to_html(data: Dict) -> str:
    """将字典渲染为一组键值卡片。"""
    if not data:
        return "<p class='meta'>（无数据）</p>"
    parts = [f"<span class='kv'>{k}：<b>{v}</b></span>" for k, v in data.items()]
    return "".join(parts)


def build_html_report(sections: Dict[str, object], title: str = "数据分析报告") -> str:
    """构建完整的 HTML 报告字符串。

    Args:
        sections: 有序字典，键为章节标题，值可为 DataFrame、dict 或字符串。
        title: 报告标题。

    Returns:
        完整的 HTML 文本，可直接写文件或供下载。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = [f"<h1>{title}</h1>", f"<p class='meta'>生成时间：{now}</p>"]

    for heading, content in sections.items():
        body.append(f"<h2>{heading}</h2>")
        if isinstance(content, pd.DataFrame):
            body.append(_df_to_html(content))
        elif isinstance(content, dict):
            body.append(_dict_to_html(content))
        else:
            body.append(f"<p>{content}</p>")

    return f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>" \
           f"<title>{title}</title>{_CSS}</head><body>" \
           f"{''.join(body)}</body></html>"
