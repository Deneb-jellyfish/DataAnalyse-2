#!/usr/bin/env bash
# Mac Jupyter 环境一键配置脚本（推荐使用 Miniforge/Conda）

set -euo pipefail

ENV_NAME="aqi-jupyter"
PYTHON_VERSION="3.11"
CONDA_BIN="${CONDA_BIN:-/opt/homebrew/Caskroom/miniforge/base/bin/conda}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "未找到 conda，请先安装 Miniforge 或 Miniconda。"
  echo "下载地址: https://github.com/conda-forge/miniforge"
  exit 1
fi

# 关闭失效代理（常见于 Clash 未启动时导致 pip/conda 失败）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="*"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> 创建/更新 conda 环境: ${ENV_NAME}"
if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "环境已存在，跳过 conda create"
else
  "$CONDA_BIN" create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" \
    jupyterlab ipykernel numpy pandas matplotlib seaborn scikit-learn \
    -c conda-forge
fi

echo "==> 安装 pip 补充依赖"
"$CONDA_BIN" run -n "$ENV_NAME" pip install chinesecalendar

echo "==> 注册 Jupyter 内核"
"$CONDA_BIN" run -n "$ENV_NAME" python -m ipykernel install --user \
  --name "$ENV_NAME" --display-name "AQI DataAnalyse-2"

echo ""
echo "配置完成。"
echo ""
echo "启动 JupyterLab:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${ROOT}"
echo "  jupyter lab"
echo ""
echo "在 Cursor/VS Code 中选择内核: AQI DataAnalyse-2"
