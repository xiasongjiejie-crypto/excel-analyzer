#!/bin/bash
# Excel 自动数据分析工具 —— 一键启动脚本（macOS）
# 双击运行：自动创建虚拟环境、安装依赖并启动应用。

# 切换到脚本所在目录（即项目根目录）
cd "$(dirname "$0")" || exit 1

echo "================================================"
echo "  📊 Excel 自动数据分析工具 启动中..."
echo "================================================"

# 1. 检测 python3
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ 未检测到 python3，请先安装 Python 3：https://www.python.org/downloads/"
    read -r -p "按回车键退出..."
    exit 1
fi

# 2. 首次运行时创建虚拟环境并安装依赖
if [ ! -d ".venv" ]; then
    echo "🔧 首次运行，正在创建虚拟环境并安装依赖（可能需要几分钟）..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# 3. 启动 Streamlit 应用（会自动打开浏览器）
echo "🚀 正在启动应用，浏览器将自动打开 http://localhost:8501"
echo "（关闭本终端窗口即可停止应用）"
streamlit run app.py

read -r -p "应用已停止，按回车键关闭窗口..."
