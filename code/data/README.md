# A 股数据同步与处理

`code/data/` 是 AITrader 的数据拉取和预处理目录。WebDAV 访问、日期筛选、本地去重和每日入口都统一走 `data/sync_ustc_webdav.py`。

数据层只负责原始数据同步、字段标准化、股票池过滤、新闻交易日对齐、样本索引和标签生成。因子构造、特征筛选、模型训练不放在这里。

命令默认在 `aitrader` conda 环境中从 `code/` 目录执行。

## 目录与脚本

| 文件 | 用途 |
| --- | --- |
| `sync_ustc_webdav.py` | WebDAV 同步脚本；支持全量缺失补齐，也支持按日期窗口增量拉取 |
| `daily_pull_cloud_data.sh` | 只拉取云端原始数据的 shell 入口，默认自动从本地最新日期的下一天开始 |
| `a_share_pipeline.py` | 读取本地原始数据，生成标准 parquet 数据表 |
| `daily_update_a_share.sh` | 先同步原始数据，再运行 `a_share_pipeline.py` |
| `load_a_share_env.sh` | 统一加载路径、账号和运行参数 |
| `run_daily_loop.sh` | 容器内常驻循环，用于执行每日任务 |
| `start_daily_loop.sh` / `stop_daily_loop.sh` | 启停常驻每日循环 |
| `install_daily_cron.sh` | 在支持 crontab 的环境里安装每日任务 |
| `a_share_pipeline.env.example` | 环境变量模板，不包含真实账号。 |
| `secrets/` | 私有密钥目录，除 `README.md` 外不纳入版本控制。 |

## 默认路径

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AITRADER_DATA_ROOT` | `data` | 数据根目录 |
| `AITRADER_RAW_DATA_DIR` | `$AITRADER_DATA_ROOT/raw_market_data` | 原始市场数据目录 |
| `AITRADER_DATASETS_ROOT` | `$AITRADER_DATA_ROOT/datasets` | 标准表、因子、特征和模型输出根目录 |
| `AITRADER_LOG_DIR` | `$AITRADER_DATA_ROOT/logs` | 日志目录 |
| `AITRADER_RUNTIME_DIR` | `$AITRADER_DATA_ROOT/runtime` | 锁、pid、运行状态目录 |
| `AITRADER_SECRETS_DIR` | `$AITRADER_DATA_ROOT/secrets` | 账号和私有配置目录 |

## 云端增量同步

最常用的方式是指定希望同步到的最新日期，`--end-date` 和 `--latest-date` 等价：

```bash
cd code
bash data/daily_pull_cloud_data.sh --end-date 2026-05-31
```

如果本地原始数据目录里最新可识别日期是 `2026-05-20`，上面的命令会只选择远端日期在 `2026-05-21` 到 `2026-05-31` 之间的文件；已经存在的本地文件会跳过。

显式指定起止日期：

```bash
python data/sync_ustc_webdav.py \
  --start-date 2026-05-21 \
  --end-date 2026-05-31
```

只指定 `--end-date` 时，脚本会扫描 `--target-dir` 下已有文件路径，找出最新日期并从下一天开始。日期从文件或目录名中识别，支持 `20260521`、`2026-05-21`、`2026/05/21`、`05-21`、`05.21` 等格式。日期窗口模式下，远端路径里没有日期的文件默认跳过；需要用 WebDAV 最后修改时间兜底时加 `--fallback-to-modified`。

全量缺失补齐仍然可用：

```bash
python data/sync_ustc_webdav.py
```

该模式会递归扫描远端目录，下载本地不存在的文件，不按日期过滤。只有显式传 `--delete` 时才会删除远端已不存在的本地文件。

## 完整日更

手动执行一次完整更新：

```bash
bash data/daily_update_a_share.sh
```

也可以把同步参数直接传给完整日更入口：

```bash
bash data/daily_update_a_share.sh --end-date 2026-05-31
```

如果想让完整日更也按日期窗口同步，可以在环境文件中设置：

```bash
A_SHARE_SYNC_END_DATE='2026-05-31'
```

或同时指定固定起点：

```bash
A_SHARE_SYNC_START_DATE='2026-05-21'
A_SHARE_SYNC_END_DATE='2026-05-31'
```

启动和停止容器内每日循环：

```bash
bash data/start_daily_loop.sh
bash data/stop_daily_loop.sh
```

## 账号与环境变量

推荐把真实账号放在私有配置目录，或通过 `A_SHARE_ENV_FILE` 指向仓库外文件：

```bash
cp data/a_share_pipeline.env.example data/secrets/a_share_pipeline.env
```

编辑 `data/secrets/a_share_pipeline.env` 并填写 WebDAV 账号。

至少需要配置：

```bash
USTC_WEBDAV_URL='https://pan.ustc.edu.cn/seafdav/'
USTC_WEBDAV_USERNAME='你的 WebDAV 用户名'
USTC_WEBDAV_PASSWORD='你的 WebDAV 密码'
```

常用可选项：

```bash
AITRADER_DATA_ROOT='data'
A_SHARE_SYNC_TARGET='data/raw_market_data'
A_SHARE_SYNC_LOG_FILE='data/logs/cloud_data_pull.log'
A_SHARE_OUTPUT_DIR='data/datasets'
A_SHARE_DAILY_TIME='08:00'
```

兼容变量：`DATASET_CLOUD_URL`、`DATASET_CLOUD_USERNAME`、`DATASET_CLOUD_PASSWORD`、`DATASET_CLOUD_TARGET`、`DATASET_CLOUD_START_DATE`、`DATASET_CLOUD_END_DATE` 仍会被 `sync_ustc_webdav.py` 读取；推荐优先使用 `A_SHARE_SYNC_*` 或 `USTC_WEBDAV_*`。

## 原始数据识别

`a_share_pipeline.py` 会在 `--raw-dir` 下递归扫描文件，并按路径关键字推断类型；也可以通过参数显式指定。

| 数据类型 | 默认识别关键字 | 显式参数 | 必要字段 |
| --- | --- | --- | --- |
| A 股量价 | `daily`、`price`、`quote`、`行情` | `--price-file` | 日期、股票代码、价格字段 |
| 基本面指标 | `metric` | `--metric-file` | 日期、股票代码 |
| 资金流 | `moneyflow` | `--moneyflow-file` | 日期、股票代码 |
| 新闻 | `news`、`article`、`report`、`新闻` | `--news-file` | 发布时间、标题或正文 |
| ST 名单 | `stock_st` | `--stock-st-file` | 日期、股票代码 |
| 股票基础信息 | 默认读取 `A股数据/basic.csv` | `--basic-file` | 股票代码 |

只运行本地预处理：

```bash
python data/a_share_pipeline.py
```

显式指定输入文件：

```bash
python data/a_share_pipeline.py \
  --price-file data/raw_market_data/A股数据/daily/20260521.csv \
  --metric-file data/raw_market_data/A股数据/metric/20260521.csv \
  --moneyflow-file data/raw_market_data/A股数据/moneyflow/20260521.csv \
  --stock-st-file data/raw_market_data/A股数据/stock_st/20260521.csv \
  --news-file data/raw_market_data/A股数据/news/20260521.csv
```

## 输出表

处理结果写入：

```text
data/datasets/processed/
```

| 文件 | 主键 | 内容 |
| --- | --- | --- |
| `basic.parquet` | `stock_code` | 股票名称、行业、市场、地区、上市日期等静态信息 |
| `price.parquet` | `stock_code + trade_date` | 日频量价表，并补齐可用静态字段 |
| `metric.parquet` | `stock_code + trade_date` | 基本面、估值、市值等指标 |
| `moneyflow.parquet` | `stock_code + trade_date` | 资金流向字段 |
| `stock_st.parquet` | `stock_code + trade_date` | ST 日频名单 |
| `news.parquet` | `trade_date + publish_time + news_text` | 新闻明细、交易日映射和股票匹配结果 |
| `news_stock_daily.parquet` | `stock_code + trade_date` | 个股新闻日聚合 |
| `news_market_daily.parquet` | `trade_date` | 市场新闻日聚合 |
| `panel.parquet` | `stock_code + trade_date` | `price`、`metric`、`moneyflow` 合并后的日频宽表 |
| `samples.parquet` | `sample_id` | 模型样本、决策时间、标签和新闻聚合字段 |
| `samples_preview.csv` | `sample_id` | 样本预览 |

每次运行会重写目标 `processed/` 和 `meta/` 下的文件。

## 数据规则

北交所过滤：代码后缀为 `.BJ`，或基础信息 `market` 包含 `北交` 的记录会从 `price`、`metric`、`moneyflow`、`panel` 中剔除。

ST 过滤：命中 `stock_st` 的 `(stock_code, trade_date)` 会从 `price`、`metric`、`moneyflow`、`panel`、`samples` 中剔除。样本表会检查 `feature_asof_date`、`target_trade_date`、`label_end_date`，任一日期命中 ST 即剔除。

新闻对齐：15:00 前发布的新闻映射到当日或之后第一个 A 股交易日；15:00 及以后发布的新闻映射到下一自然日或之后第一个 A 股交易日。

标签定义：`samples.parquet` 中一行表示某只股票在 `target_trade_date` 开盘前的一次预测事件。标签可以使用未来价格，因为标签是监督学习目标；FactorMiner 构造输入特征时必须按 `feature_asof_date` 截断，不能使用更晚的数据。
