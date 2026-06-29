# 📊 Excel 自动数据分析工具

一个基于 Streamlit 的桌面/浏览器数据分析工具：上传 Excel 文件后，自动完成
**数据画像、描述性统计、假设检验、相关性分析以及计量经济学回归与诊断**。

## ✨ 功能特性

- **上传弹窗**：点击上传按钮即可弹出系统文件选择窗口，支持 `.xlsx` / `.xls`
- **多工作表**：自动识别多个 Sheet 并可切换
- **数据画像**：自动区分数值/类别列，统计缺失率、重复行、异常值
- **描述性统计**：均值、中位数、标准差、偏度、峰度等
- **假设检验**：正态性检验、相关性分析（Pearson / Spearman）热力图
- **计量经济学**：OLS 多元回归 + 多重共线性(VIF)、异方差(BP)、自相关(DW)诊断
- **可视化**：交互式图表（Plotly）

## 🚀 快速开始

```bash
# 1. 安装依赖（建议先创建虚拟环境）
pip install -r requirements.txt

# 2. 启动应用
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，点击「上传 Excel 文件」即可。

## 📁 项目结构

```
excel_analyzer/
├── app.py                 # Streamlit 入口（上传弹窗 + 自动分析）
├── analysis/
│   ├── profiling.py       # 数据画像与质量检查
│   ├── descriptive.py     # 描述性统计
│   ├── statistical.py     # 假设检验、相关性
│   └── econometrics.py    # 回归与计量诊断
├── utils/
│   └── loader.py          # Excel 读取与清洗
├── requirements.txt
└── README.md
```

## 📝 使用建议

1. 第一行应为列名（表头）。
2. 做回归分析前，请在界面中选择「因变量 Y」和「自变量 X」。
3. 含缺失值的数据可在界面选择处理方式（删除 / 均值 / 中位数填充）。
