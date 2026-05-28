# Alpha101-polars

[English](README.md) | 中文

基于 Polars 原生实现的 WorldQuant Alpha101 因子库。

项目围绕分阶段的 `polars.Expr` 管道设计。截面算子（如 `rank(x)`）按 `date` 计算，
时序算子（如 `correlation(x, y, d)`）按 `symbol` 计算。

## 实现进度

目前已覆盖 Alpha#001 至 Alpha#101，共 82 个已注册因子：

```text
alpha001-alpha101，跳过以下 19 个需要行业中性化或市值数据的因子：
alpha048, alpha056, alpha058, alpha059, alpha063, alpha067, alpha069,
alpha070, alpha076, alpha079, alpha080, alpha082, alpha087, alpha089,
alpha090, alpha091, alpha093, alpha097, alpha100
```

每个因子函数均附带原始论文公式注释，格式统一为 `# Alpha#NNN: <公式>`。

## 数据模式

默认列定义：

```text
date, symbol, open, high, low, close, volume, vwap, returns
```

时序运算按 `symbol` 分组；截面排名按 `date` 分组。

需要市值数据或行业中性化的因子暂不注册，待 schema 扩展后启用。

## AkShare 示例数据

内置样本包含 10 只代表性 A 股：

```text
sh600519 贵州茅台    sh600036 招商银行    sh601318 中国平安
sh600030 中信证券    sh600276 恒瑞医药    sz000858 五粮液
sz000333 美的集团    sz000651 格力电器    sz300750 宁德时代
sz002475 立讯精密
```

下载 2020-2025 后复权日线数据：

```bash
uv run --extra data python examples/download_akshare_sample.py
```

数据保存至：

```text
data/raw/akshare_a_daily_hfq_2020_2025.parquet
```

## 快速开始

```python
import polars as pl
from alpha101 import compute_alphas

bars = pl.read_parquet("data/raw/akshare_a_daily_hfq_2020_2025.parquet")

# 计算指定因子
features = compute_alphas(bars, names=["alpha001", "alpha002"])

# 计算全部已注册因子
features = compute_alphas(bars)
```

## 开发

```bash
uv run --extra dev pytest
```

核心包仅依赖 Polars。AkShare 通过可选的 `data` extra 引入，用于下载 A 股示例数据。

结合截面与时序算子的公式通过分阶段计算实现：先生成临时内部列，最终仅保留请求的 alpha 列。
