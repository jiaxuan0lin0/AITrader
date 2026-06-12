# FactorMiner Core

`FactorMiner/core/` 定义因子生产的基础合同。这里不计算具体因子，也不读取业务 parquet；它只负责描述“一个因子块应该长什么样”、如何落盘、如何注册、如何把日频因子按样本可见日期对齐。

命令默认在 `aitrader` conda 环境中从 `code/` 目录执行。

## 目录职责

| 文件 | 职责 |
| --- | --- |
| `factor_spec.py` | 定义 `FactorSpec`、`FactorResult` 和多结果合并规则 |
| `factor_block.py` | 定义 `FactorBlock`，负责 parquet/manifest 原子写入 |
| `registry.py` | 维护 `factor_registry.json` / `feature_registry.json`，并校验块文件 |
| `alignment.py` | 使用 `feature_asof_date` 把日频因子对齐到 `sample_id` |

## 核心对象

### `FactorSpec`

每个因子列必须有一条 `FactorSpec` 元信息。新增因子时至少要确认：

- `name` 唯一，且和输出表列名一致。
- `source` 标记来源，例如 `alpha158`、`metric`、`moneyflow`、`news`。
- `category` 能体现因子类型，例如 `rolling.roc`、`metric.raw`。
- `inputs` 只写真实依赖字段，不写标签或未来字段。
- `expression` 能复现计算公式。
- `lookback` 是构造该因子需要回看的最长交易日数量。
- `availability` 默认保持 `feature_asof_date`。

### `FactorResult`

因子池函数应返回 `FactorResult`。标准日频主键是：

```text
stock_code, trade_date
```

样本级特征主键是：

```text
sample_id
```

写出前必须通过 `result.validate()`。它会检查主键列、主键唯一性、因子列存在性和因子名重复。

### `FactorBlock`

`FactorBlock` 是一个已落盘因子块的注册记录。它记录：

- block 名称
- 粒度：`daily` 或 `sample`
- 主键列
- parquet 路径
- manifest 路径
- 行数和因子数
- 创建时间

使用 `write_factor_block()` 生成 parquet 和 manifest，再用 `upsert_block()` 写入注册表。

## 常用流程

写出一个日频因子块：

```python
from FactorMiner.core.factor_block import write_factor_block
from FactorMiner.core.registry import upsert_block

block = write_factor_block(
    result,
    name="manual_metric",
    granularity="daily",
    factor_path="/path/to/manual_metric.parquet",
    manifest_path="/path/to/manual_metric.json",
    description="Manual daily valuation factors.",
)
upsert_block("/path/to/factor_registry.json", block)
```

校验注册表：

```bash
python -m FactorMiner.build.daily --validate-only
python -m FactorMiner.build.sample_features --validate-only
```

仅做 metadata 校验时，`registry.validate(..., metadata_only=True)` 会读取 parquet schema 和行数，不全量加载大表。

## 对齐规则

`alignment.py` 的唯一原则是：日频因子只能通过 `samples.feature_asof_date` 对齐，不能用 `target_trade_date` 当日收盘后数据。

必需输入列：

| 表 | 必需列 |
| --- | --- |
| `samples` | `sample_id`, `stock_code`, `feature_asof_date` |
| `daily_factors` | `stock_code`, `trade_date` |

对齐后保持样本原始行序。若日频因子主键重复，校验不通过。

## 修改守则

- 新增粒度时，先扩展 `GRANULARITY_KEY_COLUMNS`，再补注册表校验测试。
- 新增注册表字段时，保证 `FactorBlock.from_record()` 兼容旧记录。
- 修改 `FactorResult.validate()` 前，先检查所有 build/evaluation 流程是否依赖既有异常类型和错误信息。
- 不在 core 层加入业务字段名、路径约定或模型训练逻辑。
