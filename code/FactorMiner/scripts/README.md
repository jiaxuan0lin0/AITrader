# FactorMiner Scripts 使用手册

`FactorMiner/scripts/` 放常用工作流的 shell 封装。脚本只负责设置默认参数、切换工作目录、激活环境和写日志；核心逻辑仍在 Python 模块里。

## 脚本列表

| 脚本 | 用途 |
| --- | --- |
| `run_train_selection_slice.sh` | 使用训练期日期窗口重新聚合已有评估明细，并生成筛选结果 |
| `run_competition_selection_slice.sh` | 比赛最终版筛选封装，默认截止到 `2026-05-20` |

## 运行方式

训练期筛选：

```bash
cd code
FactorMiner/scripts/run_train_selection_slice.sh
```

比赛版筛选：

```bash
cd code
FactorMiner/scripts/run_competition_selection_slice.sh
```

脚本内部调用：

```bash
python3 -m FactorMiner.run_factor_workflow --mode select --select-engine slice
```

## 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SELECT_SINCE` | `2016-01-05` | 训练筛选起始 `target_trade_date` |
| `SELECT_UNTIL` | `2025-09-30` 或比赛脚本的 `2026-05-20` | 训练筛选截止 `target_trade_date` |
| `CORR_ROW_LIMIT` | `0` | 高相关去冗余时的相关性采样行数；`0` 表示全量 |
| `BLOCKS` | `all` | 参与筛选的 feature blocks |
| `SOURCE_EVALUATION_DIR` | `.../factors/evaluation` | 已有 `factor_ic.csv` 等明细所在目录 |
| `EVALUATION_DIR` | 按日期自动生成 | 该次筛选结果输出目录 |
| `PREPARE_REVIEW` | `1` | 是否生成 ChatGPT 复核 prompt |
| `REVIEW_PROFILE` | `research` 或 `competition` | 复核 prompt 模式 |
| `AITRADER_CODE_ROOT` | 自动推导 | `code/` 根目录 |
| `AITRADER_DATA_ROOT` | 自动推导 | `data/` 根目录 |
| `AITRADER_PYTHON_BIN` | `python3` | Python 可执行文件 |
| `CONDA_ENV` | `aitrader` | 存在 conda 初始化脚本时激活的环境 |
| `LOG_PATH` | 按日期自动生成 | 日志文件路径 |

## 示例

指定训练窗口：

```bash
SELECT_SINCE=2016-01-05 \
SELECT_UNTIL=2025-09-30 \
FactorMiner/scripts/run_train_selection_slice.sh
```

指定输出目录和复核模式：

```bash
EVALUATION_DIR=data/datasets/factors/evaluation/select_custom \
REVIEW_PROFILE=research \
FactorMiner/scripts/run_train_selection_slice.sh
```

只筛选不生成复核 prompt：

```bash
PREPARE_REVIEW=0 FactorMiner/scripts/run_train_selection_slice.sh
```

追加 Python 参数：

```bash
FactorMiner/scripts/run_train_selection_slice.sh --max-selected 200 --min-coverage 0.10
```

## 输出

默认日志：

```text
data/logs/factorminer_select_<since>_<until>_slice.log
```

默认筛选结果：

```text
<EVALUATION_DIR>/sample_feature_quality.csv
<EVALUATION_DIR>/factor_summary.csv
<EVALUATION_DIR>/selected_features.json
<EVALUATION_DIR>/review_prompt.md
<EVALUATION_DIR>/factor_workflow_summary.json
```

## 修改守则

- shell 脚本只保留薄封装，不复制 Python 业务逻辑。
- 新增环境变量时，给出默认值并同步更新本 README。
- 输出目录应带日期或业务含义，避免覆盖正式 evaluation 根目录。
- 需要改变筛选逻辑时，优先改 `FactorMiner/run_factor_workflow.py` 并补测试。
