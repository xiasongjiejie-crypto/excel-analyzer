"""Excel 自动数据分析工具 —— Streamlit 主应用。

运行方式：
    streamlit run app.py

功能流程：上传 Excel -> 选择工作表 -> 数据清洗 -> 自动分析
（数据画像 / 描述统计 / 假设检验 / 相关性 / 计量经济学回归）。
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    descriptive,
    econometrics,
    models,
    profiling,
    report,
    statistical,
)
from utils import auth, loader

st.set_page_config(page_title="Excel 自动数据分析", page_icon="📊", layout="wide")


def main() -> None:
    """应用主入口。"""
    # 访问密码校验：未通过则停止渲染后续内容
    if not auth.check_password():
        st.stop()

    st.title("📊 Excel 自动数据分析工具")
    st.caption("上传 Excel 文件，自动完成统计与计量经济学分析")

    # ---------- 侧边栏：上传与清洗配置 ----------
    with st.sidebar:
        st.header("⚙️ 数据设置")
        uploaded = st.file_uploader(
            "点击上传 Excel 文件", type=["xlsx", "xls"],
            help="点击后会弹出系统文件选择窗口，支持 .xlsx / .xls",
        )
        missing_strategy = st.selectbox(
            "缺失值处理方式",
            options=["drop", "mean", "median"],
            format_func=lambda s: {
                "drop": "删除含缺失的行",
                "mean": "数值列均值填充",
                "median": "数值列中位数填充",
            }[s],
        )
        auto_convert = st.checkbox("自动推断列类型（日期/数值）", value=True)

    if uploaded is None:
        st.info("👈 请在左侧上传一个 Excel 文件以开始分析。")
        return

    # ---------- 读取与工作表选择 ----------
    try:
        sheet_names = loader.list_sheet_names(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"无法读取该 Excel 文件：{exc}")
        return

    sheet = st.sidebar.selectbox("选择工作表", sheet_names)
    df_raw = loader.read_excel(uploaded, sheet_name=sheet)

    if auto_convert:
        df_raw = loader.infer_and_convert_types(df_raw)

    df = loader.handle_missing(df_raw, strategy=missing_strategy)

    st.success(f"✅ 读取成功：{df.shape[0]} 行 × {df.shape[1]} 列（工作表：{sheet}）")

    # ---------- 分页展示分析结果 ----------
    (tab_preview, tab_profile, tab_desc, tab_stat,
     tab_econ, tab_models, tab_report) = st.tabs(
        ["📋 数据预览", "🔍 数据画像", "📈 描述统计", "🔗 相关性检验",
         "📐 计量经济学", "🤖 更多模型", "📄 导出报告"]
    )

    with tab_preview:
        _render_preview(df)
    with tab_profile:
        _render_profiling(df)
    with tab_desc:
        _render_descriptive(df)
    with tab_stat:
        _render_statistical(df)
    with tab_econ:
        _render_econometrics(df)
    with tab_models:
        _render_models(df)
    with tab_report:
        _render_report(df, sheet)


def _render_preview(df: pd.DataFrame) -> None:
    """渲染数据预览页。"""
    st.subheader("数据预览（前 50 行）")
    st.dataframe(df.head(50), use_container_width=True)


def _render_profiling(df: pd.DataFrame) -> None:
    """渲染数据画像页。"""
    st.subheader("数据整体概览")
    ov = profiling.overview(df)
    cols = st.columns(len(ov))
    for col, (k, v) in zip(cols, ov.items()):
        col.metric(k, v)

    st.subheader("逐列画像")
    st.dataframe(profiling.column_summary(df), use_container_width=True)

    st.subheader("异常值检测（IQR 方法）")
    out = profiling.detect_outliers_iqr(df)
    if out.empty:
        st.info("没有可检测的数值列。")
    else:
        st.dataframe(out, use_container_width=True)


def _render_descriptive(df: pd.DataFrame) -> None:
    """渲染描述性统计页。"""
    st.subheader("数值列描述性统计")
    desc = descriptive.numeric_describe(df)
    if desc.empty:
        st.info("没有数值列可供统计。")
    else:
        st.dataframe(desc, use_container_width=True)

        # 分布直方图
        num_cols = desc.index.tolist()
        sel = st.selectbox("选择列查看分布", num_cols)
        fig = px.histogram(df, x=sel, marginal="box", nbins=30,
                           title=f"{sel} 的分布")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("类别列频数（Top 5）")
    cat = descriptive.categorical_summary(df)
    if not cat:
        st.info("没有类别列。")
    for col, table in cat.items():
        st.markdown(f"**{col}**")
        st.dataframe(table, use_container_width=True)


def _render_statistical(df: pd.DataFrame) -> None:
    """渲染假设检验与相关性页。"""
    st.subheader("正态性检验")
    norm = statistical.normality_test(df)
    if norm.empty:
        st.info("没有可检验的数值列。")
    else:
        st.dataframe(norm, use_container_width=True)

    st.subheader("相关性分析")
    method = st.radio("相关系数类型", ["pearson", "spearman", "kendall"],
                      horizontal=True)
    corr = statistical.correlation_matrix(df, method=method)
    if corr.empty or corr.shape[0] < 2:
        st.info("数值列不足，无法计算相关性。")
        return

    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"{method.capitalize()} 相关系数热力图")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**相关性最强的变量对**")
    st.dataframe(statistical.top_correlations(df, method=method),
                 use_container_width=True)


def _render_econometrics(df: pd.DataFrame) -> None:
    """渲染计量经济学回归页。"""
    st.subheader("OLS 多元线性回归")
    num_cols, _ = loader.split_column_types(df)
    all_cols = df.columns.tolist()

    if len(num_cols) < 1:
        st.info("没有数值列，无法进行回归分析。")
        return

    y_col = st.selectbox("选择因变量 Y（数值型）", num_cols)
    x_cols = st.multiselect(
        "选择自变量 X（可多选，支持类别变量自动编码）",
        [c for c in all_cols if c != y_col],
    )

    if not x_cols:
        st.info("请至少选择一个自变量 X。")
        return

    if not st.button("🚀 运行回归分析", type="primary"):
        return

    try:
        model = econometrics.ols_regression(df, y_col, x_cols)
    except Exception as exc:  # noqa: BLE001
        st.error(f"回归失败：{exc}")
        return

    # 模型整体指标
    st.markdown("#### 模型拟合指标")
    metrics = econometrics.model_metrics(model)
    mcols = st.columns(len(metrics))
    for col, (k, v) in zip(mcols, metrics.items()):
        col.metric(k, v)

    # 系数表
    st.markdown("#### 回归系数")
    st.caption("显著性：*** p<0.01，** p<0.05，* p<0.1")
    st.dataframe(econometrics.coefficients_table(model), use_container_width=True)

    # 诊断检验
    st.markdown("#### 多重共线性（VIF）")
    st.caption("经验上 VIF > 10 表示存在严重多重共线性")
    st.dataframe(econometrics.vif_table(df, x_cols), use_container_width=True)

    st.markdown("#### 模型诊断")
    diag = econometrics.diagnostics(model, df, y_col, x_cols)
    st.json(diag)

    # 时间序列平稳性（可选）
    with st.expander("📉 时间序列平稳性检验（ADF，可选）"):
        ts_col = st.selectbox("选择需检验的数值列", num_cols, key="adf_col")
        if st.button("运行 ADF 检验"):
            st.json(econometrics.adf_test(df[ts_col]))


def _render_models(df: pd.DataFrame) -> None:
    """渲染更多模型页（离散选择/ARIMA/聚类/PCA）。"""
    num_cols, _ = loader.split_column_types(df)
    all_cols = df.columns.tolist()

    model_choice = st.selectbox(
        "选择模型类型",
        ["Logit / Probit 离散选择", "ARIMA 时间序列预测",
         "KMeans 聚类", "PCA 主成分分析"],
    )

    # --- 离散选择 ---
    if model_choice == "Logit / Probit 离散选择":
        st.caption("因变量需为二元变量（取值 0 / 1）")
        y_col = st.selectbox("因变量 Y（0/1）", all_cols, key="dc_y")
        x_cols = st.multiselect(
            "自变量 X", [c for c in all_cols if c != y_col], key="dc_x")
        kind = st.radio("模型", ["logit", "probit"], horizontal=True)
        if x_cols and st.button("运行离散选择模型", type="primary"):
            try:
                res = models.discrete_choice(df, y_col, x_cols, model=kind)
                st.markdown("#### 系数与几率比")
                st.dataframe(models.discrete_choice_summary(res),
                             use_container_width=True)
                st.text(res.summary().as_text())
            except Exception as exc:  # noqa: BLE001
                st.error(f"模型拟合失败：{exc}")

    # --- ARIMA ---
    elif model_choice == "ARIMA 时间序列预测":
        if not num_cols:
            st.info("没有数值列。")
            return
        col = st.selectbox("选择时间序列列", num_cols, key="arima_col")
        c1, c2, c3, c4 = st.columns(4)
        p = c1.number_input("p", 0, 5, 1)
        d = c2.number_input("d", 0, 2, 1)
        q = c3.number_input("q", 0, 5, 1)
        steps = c4.number_input("预测期数", 1, 100, 10)
        if st.button("运行 ARIMA 预测", type="primary"):
            try:
                _, pred = models.arima_forecast(
                    df[col], order=(p, d, q), steps=steps)
                st.markdown("#### 预测结果")
                st.dataframe(pred, use_container_width=True)
                hist = df[col].dropna().reset_index(drop=True)
                combined = pd.concat([hist, pred["预测值"]])
                fig = px.line(combined, title=f"{col} 历史与预测")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"ARIMA 拟合失败：{exc}")

    # --- KMeans ---
    elif model_choice == "KMeans 聚类":
        feats = st.multiselect("选择聚类特征（数值）", num_cols, key="km_feat")
        k = st.slider("簇数 K", 2, 10, 3)
        if len(feats) >= 2 and st.button("运行聚类", type="primary"):
            try:
                result, counts, inertia = models.kmeans_cluster(df, feats, k)
                st.metric("簇内平方和 (Inertia)", inertia)
                st.markdown("#### 各簇样本数")
                st.dataframe(counts.rename("样本数"), use_container_width=True)
                fig = px.scatter(
                    result, x=feats[0], y=feats[1], color="cluster",
                    title="聚类结果（前两个特征）")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"聚类失败：{exc}")
        elif len(feats) < 2:
            st.info("请至少选择 2 个特征。")

    # --- PCA ---
    elif model_choice == "PCA 主成分分析":
        feats = st.multiselect("选择降维特征（数值）", num_cols, key="pca_feat")
        n = st.slider("主成分数量", 2, max(2, len(feats)) if feats else 2, 2)
        if len(feats) >= 2 and st.button("运行 PCA", type="primary"):
            try:
                scores, var_df = models.pca_analysis(df, feats, n)
                st.markdown("#### 方差解释率")
                st.dataframe(var_df, use_container_width=True)
                fig = px.scatter(scores, x="PC1", y="PC2",
                                 title="主成分散点图（PC1 vs PC2）")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"PCA 失败：{exc}")
        elif len(feats) < 2:
            st.info("请至少选择 2 个特征。")


def _render_report(df: pd.DataFrame, sheet: str) -> None:
    """渲染报告导出页，生成 HTML 报告供下载。"""
    st.subheader("导出分析报告")
    st.caption("一键汇总数据画像、描述统计、相关性等结果为 HTML 报告。")

    if not st.button("📄 生成报告", type="primary"):
        return

    # 汇总各章节内容
    sections = {
        "数据整体概览": profiling.overview(df),
        "逐列画像": profiling.column_summary(df),
        "异常值检测（IQR）": profiling.detect_outliers_iqr(df),
        "数值列描述性统计": descriptive.numeric_describe(df),
        "正态性检验": statistical.normality_test(df),
        "相关性最强变量对": statistical.top_correlations(df),
    }
    html = report.build_html_report(
        sections, title=f"数据分析报告 - {sheet}")

    st.success("✅ 报告已生成，点击下方按钮下载。")
    st.download_button(
        label="⬇️ 下载 HTML 报告",
        data=html.encode("utf-8"),
        file_name=f"analysis_report_{sheet}.html",
        mime="text/html",
    )
    with st.expander("👀 预览报告"):
        st.components.v1.html(html, height=600, scrolling=True)


if __name__ == "__main__":
    main()
