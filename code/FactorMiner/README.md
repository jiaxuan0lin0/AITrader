# FactorMiner 实现说明

`FactorMiner/` 用于在 `/data` 输出的标准数据表之上构建可训练因子、完成样本对齐并生成质量报告。`FactorMiner/` 不负责原始数据清洗、股票池过滤、`panel` 生成、`samples` 生成和标签生成；这些工作由 `/data` 模块完成。

本文定义 `FactorMiner/` 的实现流程。整体流程分为四个模块：

```text
算子模块 -> 因子模块 -> 对齐模块 -> 质检模块
```

代码分层：

```text
FactorMiner/
  operators.py   通用时序、横截面和安全数值算子
  run_pipeline.py  从 processed 数据和已缓存新闻评分开始的一键执行入口
  run_factor_workflow.py  固定因子块后的训练期筛选和推理特征组装入口
  core/        FactorSpec、FactorResult、FactorBlock、Registry、对齐规则
  build/       daily/news/sample feature 的构建入口
  pools/       人工 daily 因子池和新闻样本因子池
  news_scoring/ 新闻 LLM 打分、schema 校验和 vLLM 封装脚本
  evaluation/  质检、单因子评估、筛选结果输出
  mining/      自动化因子候选生成，不承载模型训练
  scripts/     常用筛选工作流 shell 封装
```

模块手册：

| 手册 | 内容 |
| --- | --- |
| `FactorMiner/core/README.md` | 核心合同、block、registry 和样本对齐规则 |
| `FactorMiner/build/README.md` | daily/sample/news feature 构建入口和命令 |
| `FactorMiner/pools/README.md` | 因子池输入输出合同和新增因子检查表 |
| `FactorMiner/news_scoring/README.md` | 新闻 LLM 打分服务、执行和输出字段 |
| `FactorMiner/evaluation/README.md` | 质量检查、单因子评估、筛选和复核 |
| `FactorMiner/mining/README.md` | GPT 候选因子 packet 和候选校验 |
| `FactorMiner/scripts/README.md` | 训练期和比赛版筛选脚本 |
| `FactorMiner/README_RUNBOOK.md` | 从已有数据到筛选结果的执行流程 |

模型训练不放在 `FactorMiner/` 内。FactorMiner 的边界是产出可审计、可筛选、可被下游模型消费的 feature blocks 和筛选结果。

## 1. 输入边界

`FactorMiner/` 默认读取以下标准数据表：

| 数据表 | 默认路径 | 用途 |
| --- | --- | --- |
| `price.parquet` | `data/datasets/processed/price.parquet` | 量价、波动率、流动性、趋势类因子 |
| `metric.parquet` | `data/datasets/processed/metric.parquet` | 估值、市值、换手、股息类因子 |
| `moneyflow.parquet` | `data/datasets/processed/moneyflow.parquet` | 主力资金、小单压力、资金动量类因子 |
| `news.parquet` | `data/datasets/processed/news.parquet` | 新闻明细，供大模型打分和聚合 |
| `news_stock_daily.parquet` | `data/datasets/processed/news_stock_daily.parquet` | 个股新闻日聚合数据 |
| `news_market_daily.parquet` | `data/datasets/processed/news_market_daily.parquet` | 市场新闻日聚合数据 |
| `basic.parquet` | `data/datasets/processed/basic.parquet` | 股票名称、行业、上市日期、市场等静态信息 |
| `samples.parquet` | `data/datasets/processed/samples.parquet` | 样本索引、决策时间和标签 |

`data/a_share_pipeline.py` 会把 `basic.industry` 补到 `price.parquet`、`metric.parquet` 和 `moneyflow.parquet`。FactorMiner 的中性派生层应直接消费这些标准字段，并在缺失时显式报错或关闭中性派生；单行 `industry` 缺失时，对应 `*_ind_neu` 保持缺失，不把所有缺行业股票硬合成一个行业组。

`news.parquet` 是新闻 Qwen 打分的稳定输入边界。数据层补静态字段时不应改变 `publish_time`、`trade_date`、`news_text`、`matched_stock_codes`、`matched_stock_count`，否则会影响新闻 item、股票映射和后续新闻样本因子。

`samples.parquet` 是训练样本的基准表。一行样本表示某只股票在某个 `target_trade_date` 开盘前的一次预测事件。`1/3/5/10/20/60` 等窗口不是不同样本，而是同一样本上的不同历史观察长度。

FactorMiner 必须使用 `feature_asof_date` 对齐日频因子，不能用 `target_trade_date` 当日收盘后才能知道的数据构造输入特征。

## 2. 算子模块

算子模块提供可复用的基础计算能力。所有人工定义因子和后续自动化因子挖掘都应当复用同一套算子。

实现文件：

```text
FactorMiner/operators.py
```

算子模块只负责通用计算，不包含具体因子名称，不读取 parquet，不合并样本，也不生成质量报告。

大部分时序算子接收 `DataFrame + 列名`，返回与输入 `DataFrame.index` 对齐的 `Series`。即使输入数据没有预先排序，算子也会在内部按 `stock_code + trade_date` 排序计算，并在输出时还原到原始索引顺序。

默认列名约定：

```text
股票列: stock_code
日期列: trade_date
```

可通过参数覆盖：

```text
group_col
date_col
```

### 2.1 时序算子

时序算子必须按 `stock_code` 分组，并在 `trade_date` 方向上计算。

| 算子 | 含义 |
| --- | --- |
| `delay(x, n)` | 取同一股票 n 个交易日前的值 |
| `delta(x, n)` | 当前值减去 n 个交易日前的值 |
| `returns(x, n)` | 当前值相对 n 个交易日前的收益率 |
| `ts_sum(x, n)` | 过去 n 个交易日求和 |
| `ts_mean(x, n)` | 过去 n 个交易日均值 |
| `ts_std(x, n)` | 过去 n 个交易日标准差 |
| `ts_min(x, n)` | 过去 n 个交易日最小值 |
| `ts_max(x, n)` | 过去 n 个交易日最大值 |
| `ts_quantile(x, n, q)` | 过去 n 个交易日分位数 |
| `ts_rank(x, n)` | 当前值在过去 n 个交易日窗口内的排名 |
| `ts_corr(x, y, n)` | 两个变量过去 n 个交易日滚动相关系数 |
| `ts_slope(x, n)` | 过去 n 个交易日线性趋势斜率 |
| `ts_rsquare(x, n)` | 过去 n 个交易日线性回归 R 方 |
| `ts_residual(x, n)` | 过去 n 个交易日线性回归的最新残差 |
| `ts_idxmax(x, n)` | 过去 n 个交易日最大值在窗口内的 1-based 位置 |
| `ts_idxmin(x, n)` | 过去 n 个交易日最小值在窗口内的 1-based 位置 |
| `decay_linear(x, n)` | 过去 n 个交易日线性衰减加权值 |

时序窗口默认要求完整窗口：

```text
min_periods = window
```

例如，20 日均线因子必须有完整 20 个交易日数据才会产生结果。若只对局部时间段做测试或增量计算，应当额外加载足够的历史 warmup 数据：

```text
计算所需数据区间 = 目标样本区间 + 最大 lookback 窗口
```

正式构建因子时，应先在完整历史或带 warmup 的历史数据上计算窗口因子，再截取目标样本区间。不能先截取目标区间再计算窗口，否则目标区间开头会因为人为截断历史而出现缺失。

### 2.2 横截面算子

横截面算子必须按 `trade_date` 分组，并在当日股票池内计算。

| 算子 | 含义 |
| --- | --- |
| `cs_rank(x)` | 当日横截面排名 |
| `cs_pct_rank(x)` | 当日横截面百分位排名 |
| `cs_zscore(x)` | 当日横截面标准分 |
| `industry_neutralize(x, industry)` | 按行业做横截面中性化 |

行业中性化定义为扣除同一交易日、同一行业内的均值：

```text
industry_neutralize(x) = x - mean(x | trade_date, industry)
```

该算子保留原始因子值的缺失状态，不负责填充缺失值。

### 2.3 安全算子

安全算子用于避免除零、无穷值和非法数值扩散。

| 算子 | 含义 |
| --- | --- |
| `safe_div(x, y)` | 安全除法，分母为 0 或缺失时返回缺失 |
| `safe_log(x)` | 仅对正数取对数 |
| `safe_sqrt(x)` | 仅对非负数开方 |
| `winsorize(x)` | 按分位数缩尾 |
| `clip_extreme(x)` | 按阈值裁剪极端值 |

所有算子输出都不应保留 `inf` 或 `-inf`；非法数值统一转换为缺失值。算子模块不做缺失值填充，缺失处理策略应由因子模块、质检模块或下游模型模块决定。

### 2.4 验证覆盖

算子层配有单元测试：

```text
tests/test_operators.py
```

测试覆盖以下行为：

| 检查项 | 说明 |
| --- | --- |
| 原始索引对齐 | 输入未排序时，输出仍与原始 `DataFrame.index` 对齐 |
| 股票隔离 | 时序算子不得跨股票串值 |
| 完整窗口 | rolling 算子在窗口不足时返回缺失 |
| Alpha158 补充算子 | 覆盖 `quantile`、`idxmax`、`idxmin`、`rsquare`、`residual` |
| 横截面隔离 | 横截面算子只在同一 `trade_date` 内比较 |
| 行业中性化 | 按 `trade_date + industry` 扣除组内均值 |
| 安全数值 | 除零、非法 log/sqrt 不产生无穷值 |

运行方式：

```bash
python3 -m pytest -q
```

## 3. 因子模块

因子模块负责定义和计算已有数据能够支持的基础因子池。因子应按数据源组织，但最终输出到统一的日频因子表。

因子模块必须先使用统一的元信息合同，再实现具体因子池。

实现文件：

```text
FactorMiner/core/factor_spec.py
```

### 3.1 因子元信息合同

`FactorSpec` 用于描述一个因子列的来源、公式和可见性。

| 字段 | 含义 |
| --- | --- |
| `name` | 因子列名，必须非空且唯一 |
| `source` | 数据来源，例如 `alpha158`、`metric`、`moneyflow`、`news` |
| `category` | 因子类别，例如 `kbar`、`rolling.roc`、`valuation` |
| `inputs` | 使用的原始字段 |
| `expression` | 因子表达式 |
| `window` | 历史窗口长度；无窗口因子可为空 |
| `lookback` | 计算所需最长回看长度，必须非负 |
| `availability` | 因子可见性口径，默认 `feature_asof_date` |
| `description` | 因子说明 |

`FactorResult` 用于绑定因子结果表和 `FactorSpec` 列表。

约束如下：

| 检查项 | 要求 |
| --- | --- |
| 主键列 | 因子表必须包含 `stock_code + trade_date` |
| 主键唯一性 | `stock_code + trade_date` 不得重复 |
| 因子列存在性 | 每个 `FactorSpec.name` 必须存在于因子表列中 |
| 因子名唯一性 | 同一个 `FactorResult` 内因子名不得重复 |
| 输出列 | 标准输出只包含主键列和 `FactorSpec` 声明的因子列 |

多个 `FactorResult` 可通过 `combine_factor_results` 按 `stock_code + trade_date` 外连接合并。合并时不同结果之间不得出现重复因子列。

建议目录：

```text
FactorMiner/pools/
  alpha158.py
  metric.py
  moneyflow.py
  neutral.py
  news_llm.py
```

每个因子池函数应返回 `FactorResult`，不得只返回裸 `DataFrame`。元信息用于复现、筛选、质量检查和自动化因子挖掘。

### 3.1.1 因子块和注册表

因子池函数只负责返回 `FactorResult`。实际落盘、注册、质检和后续组装由因子块协议负责。

实现文件：

```text
FactorMiner/core/factor_block.py
FactorMiner/core/registry.py
```

`FactorBlock` 描述一个已经物化的因子块，支持 `daily` 和 `sample` 两种粒度。

| 粒度 | 主键 | 适用因子 |
| --- | --- | --- |
| `daily` | `stock_code + trade_date` | Alpha158、metric、moneyflow、日频自动挖掘因子 |
| `sample` | `sample_id` | 新闻样本因子、样本级自动挖掘因子 |

`FactorBlock` 字段包括：

| 字段 | 含义 |
| --- | --- |
| `name` | 因子块名称，例如 `manual_metric` |
| `granularity` | `daily` 或 `sample` |
| `key_columns` | 因子块主键 |
| `factor_path` | parquet 输出路径 |
| `manifest_path` | JSON manifest 输出路径 |
| `factor_count` | 因子列数量 |
| `row_count` | 行数 |
| `created_at` | 构建时间 |
| `description` | 说明 |

`write_factor_block(...)` 用于把一个 `FactorResult` 写成 parquet 和 manifest，并返回对应 `FactorBlock`。

`FactorRegistry` 用于管理 `factor_registry.json`，提供：

```text
load
save
upsert
validate
```

注册表校验内容：

| 检查项 | 要求 |
| --- | --- |
| block 路径 | parquet 和 manifest 必须存在 |
| 主键 | 与 `granularity` 一致，且 parquet 内主键唯一 |
| manifest | manifest 中所有 `FactorSpec.name` 必须存在于 parquet |
| 行数和因子数 | `row_count`、`factor_count` 必须与实际文件一致 |
| 因子名 | 不同 block 之间不得出现重复因子列名 |

测试文件：

```text
tests/test_factor_block_registry.py
```

测试覆盖：

| 检查项 | 说明 |
| --- | --- |
| block 物化 | `write_factor_block` 输出 parquet 和 manifest |
| 粒度主键 | daily/sample 主键不匹配时报错 |
| registry 路径 | 支持相对路径校验 |
| upsert | 同名 block 会被替换 |
| 重复因子 | 不同 block 重复因子名时报错 |
| manifest 缺列 | manifest 声明了 parquet 中不存在的因子时报错 |

运行方式：

```bash
python3 -m pytest tests/test_factor_block_registry.py -q
python3 -m pytest -q
```

### 3.1.2 Daily 构建入口

日频因子块构建入口：

```text
FactorMiner/build/daily.py
```

它负责读取 `data/datasets/processed/` 下的 `price.parquet`、`metric.parquet`、`moneyflow.parquet` 和 `basic.parquet`，构建并注册：

```text
manual_alpha158_kbar
manual_alpha158_price
manual_alpha158_return
manual_alpha158_rolling3
manual_alpha158_rolling5
manual_alpha158_rolling10
manual_alpha158_rolling20
manual_alpha158_rolling60
manual_metric
manual_moneyflow
```

`alpha158` 默认用 `--alpha-layout split` 拆成多个物理 block，因子列名仍然统一使用 `alpha158_` 前缀。这样可以降低单个 parquet 的内存峰值，也能在日志中看到具体 alpha 子块进度。需要单文件输出时可显式传 `--alpha-layout single`，输出 `manual_alpha158`。

默认输出：

```text
data/datasets/factors/blocks/daily/
data/datasets/factors/manifests/
data/datasets/factors/factor_registry.json
```

小范围验证：

```bash
python3 -m FactorMiner.build.daily --block all --since 2019-01-01 --until 2019-01-31 --stock-limit 10 --output-root /tmp/factor_daily_check --registry-path /tmp/factor_daily_check/factor_registry.json
python3 -m FactorMiner.build.daily --validate-only --registry-path /tmp/factor_daily_check/factor_registry.json
```

`--since` 只过滤最终输出，不截断计算历史；这样 `ma20`、`ROC60`、`delta10` 这类窗口因子可以使用目标日期之前的历史数据。

`--stock-limit` 只能用于小范围验证，且必须写到非默认 `--output-root`，避免把小股票池横截面结果误写进正式因子库。

`--alpha-layout split` 是正式默认值；`--alpha-layout single` 只用于需要 `manual_alpha158.parquet` 单文件命名的兼容场景。

### 3.1.3 样本对齐

日频因子对齐工具：

```text
FactorMiner/core/alignment.py
FactorMiner/build/sample_features.py
```

`align_daily_factors_to_samples(samples, daily_factors)` 只使用：

```text
samples.stock_code + samples.feature_asof_date
= daily.stock_code + daily.trade_date
```

不会使用 `target_trade_date` 合并日频因子，避免把目标交易日收盘后才知道的数据泄露进开盘前样本。

`FactorMiner/build/sample_features.py` 会读取 daily `factor_registry.json`，把选中的 daily block 对齐成 `sample_id` 粒度的 feature block：

```text
data/datasets/features/blocks/sample/*_sample.parquet
data/datasets/features/manifests/*_sample.json
data/datasets/features/feature_registry.json
```

小范围验证：

```bash
python3 -m FactorMiner.build.sample_features \
  --source-registry-path /tmp/factor_daily_check/factor_registry.json \
  --output-root /tmp/sample_features_check \
  --feature-registry-path /tmp/sample_features_check/feature_registry.json \
  --blocks manual_metric \
  --since 2019-01-03 --until 2019-01-31 --limit 1000 \
  --validate-source
```

### 3.2 中性派生层

中性派生层实现文件：

```text
FactorMiner/pools/neutral.py
```

`NeutralConfig` 控制是否生成以下派生因子：

| 后缀 | 含义 |
| --- | --- |
| `_cs_pct` | 当日横截面百分位 |
| `_cs_z` | 当日横截面 zscore |
| `_ind_neu` | 同日同行业均值中性化残差 |

中性派生层不改变原始因子列，只追加派生列，并为每个派生列生成 `FactorSpec`。

如果 `industry` 为空，`*_ind_neu` 返回缺失。`metric` 因子池还会用 `MetricConfig.max_neutral_missing_rate` 控制中性派生候选，默认不对缺失率过高的基础因子追加 `cs/ind_neu` 派生列。

### 3.3 Alpha158 改造版量价因子

Alpha158 改造版实现文件：

```text
FactorMiner/pools/alpha158.py
```

量价因子使用 `price.parquet` 构造。核心字段包括：

```text
open, high, low, close, preclose, vwap, volume, amount
```

基础公式族参考 Qlib Alpha158，窗口按比赛周期调整。

默认窗口配置：

```text
return_windows  = 1, 3, 5, 10, 20, 60
rolling_windows = 3, 5, 10, 20, 60
price_windows   = 0, 1, 2, 3
```

命名统一使用 `alpha158_` 前缀，例如：

```text
alpha158_KMID
alpha158_ROC1
alpha158_MA3
alpha158_CORR10
```

覆盖以下类别：

| 类别 | 示例因子 |
| --- | --- |
| K 线结构 | `KMID`、`KLEN`、`KMID2`、`KUP`、`KUP2`、`KLOW`、`KLOW2`、`KSFT`、`KSFT2` |
| 价格归一化 | `OPEN0`、`HIGH0`、`LOW0`、`VWAP0` 及对应 delay 窗口 |
| 收益变化 | `ROC1`、`ROC3`、`ROC5`、`ROC10`、`ROC20`、`ROC60` |
| 趋势和波动 | `MA`、`STD`、`BETA`、`RSQR`、`RESI` |
| 高低位和分位 | `MAX`、`MIN`、`QTLU`、`QTLD`、`RANK`、`RSV` |
| 位置类 | `IMAX`、`IMIN`、`IMXD` |
| 量价关系 | `CORR`、`CORD`、`CNTP`、`CNTN`、`CNTD` |
| 价格涨跌强度 | `SUMP`、`SUMN`、`SUMD` |
| 成交量强度 | `VMA`、`VSTD`、`WVMA`、`VSUMP`、`VSUMN`、`VSUMD` |

`1/3/5/10/20/60` 等窗口只表示同一样本上的不同历史观察长度，不改变样本粒度。

### 3.4 估值、市值和换手因子

估值类因子使用 `metric.parquet` 构造。可用字段以实际数据表为准，常见字段包括：

```text
turnover_rate, turnover_rate_f, volume_ratio,
pe, pe_ttm, pb, ps, ps_ttm,
dv_ratio, dv_ttm,
total_mv, circ_mv
```

建议覆盖以下类别：

| 类别 | 示例因子 |
| --- | --- |
| 估值水平 | `metric_pe_ttm`、`metric_pb`、`metric_ps_ttm`、`metric_dv_ttm` |
| 估值反向 | `metric_earnings_yield`、`metric_book_to_price`、`metric_sales_to_price` |
| 市值暴露 | `metric_log_total_mv`、`metric_log_circ_mv` |
| 交易热度 | `metric_turnover_rate`、`metric_volume_ratio` |
| 变化趋势 | `metric_pe_ttm_deltaN`、`metric_pb_deltaN`、`metric_turnover_rate_maN` |
| 中性派生 | `metric_pe_ttm_cs_pct`、`metric_pb_cs_z`、`metric_total_mv_ind_neu` |

估值和市值因子应保留原始值，同时可输出横截面分位或横截面标准分。原始值不应被覆盖。

### 3.5 资金流因子

资金流因子使用 `moneyflow.parquet`，并与 `price.amount` 对齐后构造比例因子。

资金流金额字段的单位为万元，`price.amount` 的单位为千元。计算金额比例时必须先将资金流金额乘以 10：

```text
moneyflow_amount_in_price_unit = moneyflow_amount * 10
ratio = moneyflow_amount_in_price_unit / price.amount
```

建议覆盖以下类别：

| 类别 | 示例因子 |
| --- | --- |
| 总净流入 | `mf_net_amount_ratio`、`mf_net_vol_ratio` |
| 主力净流入 | `mf_main_net_amount_ratio`、`mf_main_net_amount_ratio_ma_N` |
| 大单参与度 | `mf_large_order_amount_ratio`、`mf_elg_order_amount_ratio` |
| 小单压力 | `mf_small_order_pressure` |
| 资金动量 | `mf_net_amount_ratio_maN`、`mf_net_amount_ratio_slopeN`、`mf_main_positive_daysN` |
| 量价资金共振 | `mf_price_flow_confirmN`、`mf_price_flow_corrN` |
| 中性派生 | `mf_main_net_amount_ratio_cs_pct`、`mf_main_net_amount_ratio_ind_neu` |

主力资金建议定义为大单和特大单：

```text
buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount
```

### 3.6 新闻大模型样本因子

新闻因子不输出为普通日频表。新闻具有精确发布时间，必须按样本的 `decision_ts` 判断可见性，因此新闻因子直接构建为 `sample_id` 粒度的样本因子块。

实现文件：

```text
FactorMiner/pools/news_llm.py
FactorMiner/pools/news_sample.py
FactorMiner/build/news_sample.py
```

处理流程：

```text
news.parquet
-> prepare_news_items(news)
-> news_llm_scores.parquet
-> build_news_sample_factors(samples, news_items, news_stock_map, news_scores)
-> data/datasets/features/blocks/sample/news_llm_market_sample.parquet
-> data/datasets/features/blocks/sample/news_llm_stock_sample.parquet
-> data/datasets/features/feature_registry.json
```

输入表：

| 表 | 默认路径 | 用途 |
| --- | --- | --- |
| `samples.parquet` | `data/datasets/processed/samples.parquet` | 提供 `sample_id`、`stock_code`、`decision_ts` |
| `news.parquet` | `data/datasets/processed/news.parquet` | 新闻明细和股票匹配字段 |
| `news_llm_scores.parquet` | `data/datasets/factors/news_llm_scores.parquet` | 单条新闻 LLM 评分缓存 |

输出表：

```text
data/datasets/features/blocks/sample/news_llm_market_sample.parquet
data/datasets/features/blocks/sample/news_llm_stock_sample.parquet
```

输出主键：

```text
sample_id
```

对应 manifest：

```text
data/datasets/features/manifests/news_llm_market_sample.json
data/datasets/features/manifests/news_llm_stock_sample.json
```

对应 registry：

```text
data/datasets/features/feature_registry.json
```

新闻可见性规则：

```text
window_start < publish_time <= decision_ts
```

新闻窗口与其他因子窗口体系保持一致，默认使用：

```text
1d, 3d, 5d, 10d
```

新闻窗口是自然日窗口，不是交易日窗口：

```text
1d  = decision_ts - 24h  到 decision_ts
3d  = decision_ts - 72h  到 decision_ts
5d  = decision_ts - 120h 到 decision_ts
10d = decision_ts - 240h 到 decision_ts
```

市场级新闻和个股级新闻分别聚合：

| 类型 | 识别方式 | 输出前缀 | 合并口径 |
| --- | --- | --- | --- |
| 市场级新闻 | `matched_stock_count == 0` | `news_market_*` | 同一 `decision_ts` 下样本共享 |
| 个股级新闻 | `news_stock_map` 中存在 `news_id + stock_code` | `news_stock_*` | 按 `stock_code + decision_ts` 聚合 |

新闻原文和摘要不进入训练表。训练表只保留聚合后的数值因子。`summary` 只用于人工审计和评分质量检查。

每个窗口输出以下因子族：

| 因子族 | 示例 |
| --- | --- |
| 覆盖度 | `news_market_count_1d`、`news_stock_count_3d` |
| 强事件 | `news_market_high_impact_count_1d`、`news_stock_negative_high_impact_count_5d` |
| 均值状态 | `news_market_sentiment_mean_3d`、`news_stock_risk_mean_10d` |
| 加权情绪 | `news_market_impact_weighted_sentiment_3d`、`news_stock_relevance_weighted_sentiment_5d` |
| 尾部风险 | `news_market_max_risk_1d`、`news_stock_min_sentiment_3d` |
| 最新事件 | `news_stock_latest_sentiment_1d`、`news_stock_hours_since_latest_10d` |
| 事件结构 | `news_market_policy_count_3d`、`news_stock_earnings_count_5d` |

默认事件类型结构：

| 前缀 | 事件类型 |
| --- | --- |
| `news_market_*` | `policy`、`macro`、`rates`、`fx`、`geopolitics`、`commodity` |
| `news_stock_*` | `company`、`earnings`、`litigation`、`contract` |

构建命令：

```bash
python3 -m FactorMiner.build.news_sample
```

默认会分块输出 `news_llm_market_sample` 和 `news_llm_stock_sample`，避免全量时把 `news_market_*` 和 `news_stock_*` 合成一个宽表。需要单文件输出时可显式传：

```bash
python3 -m FactorMiner.build.news_sample --scope all
```

小样本验证命令：

```bash
python3 -m FactorMiner.build.news_sample \
  --since 2019-01-16 --until 2019-01-16 --limit 200 \
  --output-root /tmp/news_sample_check \
  --feature-registry-path /tmp/news_sample_check/feature_registry.json
```

自定义窗口：

```bash
python3 -m FactorMiner.build.news_sample --windows 1,3,5,10
```

## 4. 对齐模块

对齐模块负责把日频因子和新闻因子合并到 `samples.parquet`。

实现文件：

```text
FactorMiner/core/alignment.py
FactorMiner/build/sample_features.py
```

### 4.1 日频因子对齐

日频因子表主键为：

```text
stock_code + trade_date
```

样本表主键为：

```text
sample_id
```

日频因子必须按以下规则对齐：

```text
samples.stock_code + samples.feature_asof_date
=
daily_factors.stock_code + daily_factors.trade_date
```

对齐后，一行样本仍然只表示一次预测事件。不同窗口因子应作为不同列存在于同一行样本中。

### 4.2 新闻因子对齐

新闻因子必须使用 `decision_ts` 过滤可见新闻：

```text
publish_time <= decision_ts
```

个股新闻因子按 `stock_code` 聚合，市场新闻因子不要求 `stock_code`。若新闻只包含市场级信息，则只能生成市场级新闻因子。

### 4.3 输出样本特征块

默认输出分块 sample feature，不提前生成巨型总表：

```text
data/datasets/features/blocks/sample/
data/datasets/features/feature_registry.json
```

每个 sample feature block 至少包含：

```text
sample_id
factor columns
```

训练前的总表组装留到 feature set 和 quality 阶段，按 `sample_id` 按需合并候选特征。标签列仍只来自 `/data` 生成的 `samples.parquet`，FactorMiner 不重新生成标签，也不在本模块内训练最终模型。

## 5. 质检模块

质检模块负责验证因子表和样本特征表是否可用于训练和评估。

实现文件：

```text
FactorMiner/evaluation/quality.py
```

### 5.1 必检项目

| 检查项 | 要求 |
| --- | --- |
| 主键重复 | `daily_factors` 中 `stock_code + trade_date` 不得重复，`sample_features` 中 `sample_id` 不得重复 |
| 字段缺失 | 必要输入字段缺失时应报错 |
| 无穷值 | 因子列不得保留 `inf` 或 `-inf` |
| 覆盖率 | 每个因子应统计非缺失比例 |
| 常数列 | 全部取值相同的因子应被标记 |
| 极端值 | 每个因子应统计分位数和异常范围 |
| 时间穿越 | 日频因子必须来自 `feature_asof_date`，新闻必须满足 `publish_time <= decision_ts` |
| 单位处理 | 资金流金额比例必须确认已完成单位换算 |

### 5.2 输出文件

质检模块应输出：

```text
data/datasets/factors/evaluation/sample_feature_quality.csv
data/datasets/factors/evaluation/sample_feature_block_quality.csv
data/datasets/factors/evaluation/sample_feature_quality_summary.json
```

`sample_feature_quality.csv` 记录每个 sample 级因子的覆盖率、缺失率、无穷值数量、常数列标记、分位数和按年覆盖率。

`sample_feature_block_quality.csv` 记录每个 sample feature block 的行数、样本匹配率和额外/缺失样本数量。

运行方式：

```bash
python3 -m FactorMiner.evaluation.quality
```

### 5.3 单因子评估

单因子评估模块按 `sample_id` 合并样本标签和 sample feature block，并在每个 `target_trade_date` 横截面上计算 IC、RankIC 和分组收益。

实现文件：

```text
FactorMiner/evaluation/single_factor.py
```

默认标签：

```text
label_next_open_return
label_next_vwap_return
```

输出：

```text
data/datasets/factors/evaluation/factor_ic.csv
data/datasets/factors/evaluation/factor_rankic.csv
data/datasets/factors/evaluation/group_return.csv
data/datasets/factors/evaluation/factor_summary.csv
```

运行方式：

```bash
python3 -m FactorMiner.evaluation.single_factor
```

### 5.4 自动筛选

自动筛选模块读取质量报告和单因子评估报告，先生成一版可直接给下游模型使用的自动因子列表，再额外输出高相关冲突、相关性聚类和可选复核包。

实现文件：

```text
FactorMiner/evaluation/selection.py
```

输出：

```text
data/datasets/factors/evaluation/selected_features.csv
data/datasets/factors/evaluation/final/selected_features.json
data/datasets/factors/evaluation/candidate_features.csv
data/datasets/factors/evaluation/rejected_features.csv
data/datasets/factors/evaluation/correlation_conflicts.csv
data/datasets/factors/evaluation/correlation_clusters.csv
data/datasets/factors/evaluation/review_packet.json
```

`selected_features.json` 是自动版可用因子清单。`review_packet.json` 是可选人工或 ChatGPT 复核材料，不阻塞自动流程。

运行方式：

```bash
python3 -m FactorMiner.evaluation.selection
```

### 5.5 可选 ChatGPT 复核

复核模块面向没有 API key、但可以使用 ChatGPT 订阅网页的流程：先生成一份可直接粘贴的审查 prompt 和 JSON 模板；你把 ChatGPT 返回的 JSON 保存回来后，再应用成 reviewed 版本因子清单。

实现文件：

```text
FactorMiner/evaluation/review_selection.py
```

输出：

```text
data/datasets/factors/evaluation/review_prompt.md
data/datasets/factors/evaluation/review_inputs.txt
data/datasets/factors/evaluation/review_response_template.json
data/datasets/factors/evaluation/final/selected_features_reviewed.json
data/datasets/factors/evaluation/selected_features_reviewed.csv
data/datasets/factors/evaluation/selection_review_audit.csv
data/datasets/factors/evaluation/selection_review_report.md
```

运行方式：

```bash
python3 -m FactorMiner.evaluation.review_selection --prepare
# 将 review_prompt.md 粘贴给 ChatGPT，把返回 JSON 保存为 review_response.json
python3 -m FactorMiner.evaluation.review_selection --apply --response-path data/datasets/factors/evaluation/review_response.json
```

`selected_features_reviewed.json` 是复核后的最终清单；如果不做人工复核，下游直接使用 `selected_features.json`。

### 5.6 固定因子块后的 workflow

当 daily block、sample feature block 和新闻样本因子已经构建完成后，后续不需要重复刷新因子块，可以使用：

```text
FactorMiner/run_factor_workflow.py
```

按训练日期重新筛选因子：

```bash
FactorMiner/scripts/run_train_selection_slice.sh
```

正式比赛最终版：

```bash
FactorMiner/scripts/run_competition_selection_slice.sh
```

等价命令：

```bash
python3 -m FactorMiner.run_factor_workflow \
  --mode select \
  --select-engine slice \
  --select-since 2016-01-05 \
  --select-until 2025-09-30 \
  --review-profile research \
  --prepare-review
```

按目标交易日组装模型推理特征：

```bash
python3 -m FactorMiner.run_factor_workflow \
  --mode inference \
  --target-date 2026-05-20 \
  --selected-features-path data/datasets/factors/evaluation/final/selected_features_reviewed.json
```

`select` 只用 `--select-since/--select-until` 指定的训练窗口做质量检查和筛选。`--select-engine slice` 会从已有 `factor_ic.csv`、`factor_rankic.csv`、`group_return.csv` 按训练日期重聚合 `factor_summary.csv`，避免重新跑最慢的单因子全计算。`--review-profile research/competition` 会生成不同用途的 AI 复核 prompt；`inference` 只读取 `target_trade_date == --target-date` 的样本，默认不输出 `label_` 列。

### 5.7 GPT 候选因子挖掘 packet

模块一只生成给 GPT 的小型材料包，不重算因子、不重新新闻打分、不修改已筛选因子：

```bash
python3 -m FactorMiner.mining.build_packet \
  --profile competition \
  --cutoff-date 2026-05-20 \
  --round-name next_regime_20260520 \
  --candidate-count 150
```

输出目录：

```text
data/datasets/factors/gpt_mining/experiment/next_regime_20260520/packet/
```

先读 `gpt_inputs.txt`，按里面的“建议上传”把 `prompt_generate_candidates.md`、`candidate_schema.json`、字段表、已有因子摘要、新闻字段说明和泄露规则提供给 GPT。不要上传原始 parquet 大表。prompt 会强制 GPT 先联网调研 cutoff 前 A 股市场结构、市场情绪、资金偏好、行业主线和前沿因子设计，再输出候选；但不要求每个候选逐条绑定网页来源。详细说明见 `FactorMiner/mining/README.md`。

GPT 返回后保存为轮次根目录的 `gpt_response.json`，然后运行模块二候选校验：

```bash
python3 -m FactorMiner.mining.validate_candidates \
  --round-dir data/datasets/factors/gpt_mining/experiment/next_regime_20260520
```

校验输出在：

```text
data/datasets/factors/gpt_mining/experiment/next_regime_20260520/validated/
```

当前最终采用的 GPT-mining 版本固定在：

```text
data/datasets/factors/gpt_mining/final/
```

## 6. 推荐输出结构

FactorMiner 的输出目录建议分成两个边界：

```text
data/datasets/factors/
data/datasets/features/
```

推荐输出文件：

| 文件 | 内容 |
| --- | --- |
| `blocks/daily/*.parquet` | `stock_code + trade_date` 粒度的日频因子块 |
| `data/datasets/features/blocks/sample/*.parquet` | `sample_id` 粒度的样本特征块 |
| `factor_registry.json`、`feature_registry.json` | 因子块和样本特征块注册表 |
| `manifests/*.json` | 因子定义和元信息 |
| `evaluation/sample_feature_quality.csv` | 样本特征质量报告 |
| `evaluation/sample_feature_block_quality.csv` | 样本特征块质量报告 |
| `evaluation/factor_summary.csv` | 单因子 IC、RankIC 和分组收益汇总 |
| `evaluation/selected_features.json` | 自动筛选后的可用因子清单 |
| `evaluation/review_packet.json` | 可选复核材料 |
| `evaluation/selected_features_reviewed.json` | 可选复核后的最终因子清单 |
| `news_llm_scores.parquet` | 新闻大模型打分缓存 |

## 7. 实现约束

- FactorMiner 不应修改 `/data` 的标准输出表。
- FactorMiner 不应重新生成 `panel`、`samples` 或标签。
- 所有训练输入因子必须在样本 `decision_ts` 时点可见。
- 日频因子必须通过 `feature_asof_date` 对齐样本。
- 新闻因子必须通过 `publish_time <= decision_ts` 对齐样本。
- `1/3/5/10/20/60` 等窗口只表示历史观察长度，不改变样本粒度。
- 主标签保持为下一交易日收益，不扩展 3/5/10 日持有期标签。
- 自动化因子挖掘应复用同一套算子、因子元信息、对齐逻辑和质检逻辑。
