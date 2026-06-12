import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
EXP_DIR = ROOT.parent / "data" / "experiments" / "msgca"


FIGURE_FONT = "DejaVu Sans"
for font_path in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
):
    path = Path(font_path)
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        FIGURE_FONT = font_manager.FontProperties(fname=str(path)).get_name()

plt.rcParams.update(
    {
        "font.family": FIGURE_FONT,
        "font.sans-serif": [FIGURE_FONT, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D2D2D7",
        "axes.labelcolor": "#1D1D1F",
        "axes.titlecolor": "#1D1D1F",
        "xtick.color": "#6E6E73",
        "ytick.color": "#6E6E73",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.5,
        "axes.titleweight": "semibold",
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.2,
    }
)


COLORS = {
    "ink": "#1D1D1F",
    "muted": "#6E6E73",
    "rule": "#D2D2D7",
    "blue": "#0071E3",
    "green": "#2F7D55",
    "orange": "#B56A1C",
    "red": "#A13D3A",
    "purple": "#6E5AA8",
    "gray": "#8E8E93",
    "light_blue": "#F2F8FF",
    "light_green": "#F3FAF6",
    "light_orange": "#FFF7ED",
    "light_purple": "#F7F4FF",
    "light_gray": "#F5F5F7",
}


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    value = row.get(key, "")
    if value in ("", "未记录", None):
        return default
    return float(value)


def first_row(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    raise KeyError(filters)


def box(ax, xy, wh, text, fc, ec="#D2D2D7", fontsize=10):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=0.75,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"])
    return patch


def arrow(ax, start, end, color="#8E8E93", lw=1.05):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=2, shrinkB=2),
    )


def workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    ax.set_axis_off()
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.2)

    top = [
        ("原始表\n行情/新闻/因子", COLORS["light_gray"]),
        ("可见性\n对齐", COLORS["light_blue"]),
        ("因子块\n人工+GPT", COLORS["light_green"]),
        ("样本矩阵\nas-of 特征", COLORS["light_green"]),
        ("MSGCA\n训练", COLORS["light_purple"]),
        ("每日分数\nTopN 组合", COLORS["light_orange"]),
    ]
    xs = np.linspace(0.35, 10.85, len(top))
    for i, ((label, color), x) in enumerate(zip(top, xs)):
        box(ax, (x, 2.55), (1.45, 0.78), label, color, fontsize=9)
        if i > 0:
            arrow(ax, (xs[i - 1] + 1.45, 2.94), (x, 2.94))

    lower = [
        ("防泄露边界\nfeature_asof_date", (1.6, 1.2), "#FFF6E6", "#B87923"),
        ("训练/验证/留出\n按日历切分", (4.0, 1.2), "#F2F6FA", "#5C6D7B"),
        ("Rolling-10 指标\n收益/超额/MDD", (6.6, 1.2), "#F2F6FA", "#5C6D7B"),
        ("实验归档\nruns/eval/final_selected", (9.2, 1.2), "#F4F4F4", "#69737D"),
    ]
    for label, pos, fc, ec in lower:
        box(ax, pos, (1.85, 0.72), label, fc, ec=ec, fontsize=8.4)
    arrow(ax, (2.52, 1.92), (2.52, 2.55), color="#B87923", lw=1.15)
    arrow(ax, (4.93, 1.92), (5.0, 2.55), color="#5C6D7B", lw=1.15)
    arrow(ax, (7.52, 1.92), (8.0, 2.55), color="#5C6D7B", lw=1.15)
    arrow(ax, (10.12, 1.92), (11.58, 2.55), color="#69737D", lw=1.15)

    ax.text(
        6.25,
        0.35,
        "流程将数据可见性、模型训练、策略评估和预测产物归档分离记录。",
        ha="center",
        va="center",
        fontsize=9,
        color="#48525C",
    )
    save(fig, "workflow_overview")


def msgca_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    ax.set_axis_off()
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6.1)

    ax.text(0.15, 5.75, "输入", fontsize=10, weight="bold", color="#334")
    ax.text(3.0, 5.75, "Token 编码器", fontsize=10, weight="bold", color="#334")
    ax.text(6.1, 5.75, "融合", fontsize=10, weight="bold", color="#334")
    ax.text(9.0, 5.75, "多头输出与目标", fontsize=10, weight="bold", color="#334")

    inputs = [
        ("价格窗口\nOHLCV/VWAP/资金流", (0.35, 4.55), COLORS["light_blue"]),
        ("新闻状态\nLLM 评分/数量", (0.35, 3.35), COLORS["light_orange"]),
        ("入选因子\n人工+挖掘", (0.35, 2.15), COLORS["light_green"]),
        ("上下文特征\n主题+动态聚类", (0.35, 0.95), COLORS["light_purple"]),
    ]
    encoders = [
        ("RevIN+\n时序编码", (2.9, 4.55), "#F8FBFD"),
        ("文本数值\n投影", (2.9, 3.35), "#FFF9F2"),
        ("因子分组\n投影", (2.9, 2.15), "#F3FAF5"),
        ("上下文\n投影", (2.9, 0.95), "#F6F3FB"),
    ]
    tokens = [
        ("价格 token", (4.75, 4.66)),
        ("新闻 token", (4.75, 3.46)),
        ("因子 tokens", (4.75, 2.26)),
        ("上下文 token", (4.75, 1.06)),
    ]

    for (label, pos, color), (enc, epos, ecolor), (tok, tpos) in zip(inputs, encoders, tokens):
        box(ax, pos, (1.65, 0.72), label, color, fontsize=8.7)
        box(ax, epos, (1.45, 0.72), enc, ecolor, fontsize=8.5)
        box(ax, tpos, (1.18, 0.5), tok, "#FFFFFF", fontsize=8.3)
        arrow(ax, (pos[0] + 1.65, pos[1] + 0.36), (epos[0], epos[1] + 0.36), lw=1.25)
        arrow(ax, (epos[0] + 1.45, epos[1] + 0.36), (tpos[0], tpos[1] + 0.25), lw=1.25)

    box(ax, (6.25, 3.55), (1.55, 0.72), "模态门控", COLORS["light_gray"], fontsize=8.8)
    box(ax, (6.25, 2.35), (1.55, 0.82), "Cross-attention\n融合", COLORS["light_purple"], fontsize=8.8)
    box(ax, (6.25, 1.25), (1.55, 0.72), "共享隐表示", "#FFFFFF", fontsize=8.5)
    for _, tpos in tokens:
        arrow(ax, (tpos[0] + 1.18, tpos[1] + 0.25), (6.25, 2.76), lw=1.1)
    arrow(ax, (7.02, 3.55), (7.02, 3.17), lw=1.15)
    arrow(ax, (7.02, 2.35), (7.02, 1.97), lw=1.15)

    heads = [
        ("final_score\n排序头", (8.85, 4.25), COLORS["light_blue"]),
        ("return_pred\n收益头", (8.85, 3.05), COLORS["light_green"]),
        ("direction_prob\n方向头", (8.85, 1.85), COLORS["light_orange"]),
    ]
    for label, pos, color in heads:
        box(ax, pos, (1.55, 0.72), label, color, fontsize=8.5)
        arrow(ax, (7.8, 1.61), (pos[0], pos[1] + 0.36), lw=1.1)

    losses = [
        ("Lambda@20/40\nSoftTopK@10/20/40", (10.85, 4.25)),
        ("Huber/MSE\nopen+VWAP 收益", (10.85, 3.05)),
        ("Masked BCE\n多头一致性", (10.85, 1.85)),
    ]
    for label, pos in losses:
        box(ax, pos, (1.28, 0.72), label, "#FFFFFF", fontsize=7.7)
        arrow(ax, (10.4, pos[1] + 0.36), (pos[0], pos[1] + 0.36), lw=1.0)

    box(
        ax,
        (0.35, 5.72),
        (5.1, 0.05),
        "",
        "#FFF6E6",
        ec="#FFF6E6",
        fontsize=1,
    )
    ax.text(
        0.35,
        5.46,
        "可见性边界：feature_asof_date 与新闻 publish_time 均早于决策时间。",
        fontsize=8.3,
        color="#6B4B14",
    )
    save(fig, "msgca_architecture")


def cluster_context_framework() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)

    columns = [
        ("每日股票池", "价格/因子/新闻\n决策时可见", 0.35, COLORS["light_gray"]),
        ("主题特征", "动量/资金流\nstrict context 字段", 2.55, COLORS["light_green"]),
        ("动态聚类", "股票级上下文\n簇编号+强度", 4.75, COLORS["light_purple"]),
        ("MSGCA 训练", "上下文 token\n聚类感知 loss", 6.95, COLORS["light_blue"]),
        ("Top20 组合", "direct_theme_soft\n排序输出", 9.15, COLORS["light_orange"]),
    ]
    for i, (title, body, x, color) in enumerate(columns):
        box(ax, (x, 2.35), (1.72, 0.92), f"{title}\n{body}", color, fontsize=8.2)
        if i > 0:
            arrow(ax, (columns[i - 1][2] + 1.72, 2.81), (x, 2.81), lw=1.35)

    box(ax, (4.05, 0.85), (2.95, 0.75), "聚类上下文：\ncluster_size / strength / mf / hp", "#FFFFFF", fontsize=8.1)
    arrow(ax, (5.48, 1.6), (5.6, 2.35), color=COLORS["purple"], lw=1.2)

    box(ax, (6.75, 0.55), (3.3, 0.9), "训练约束：\ncluster_topk_return + cluster_rank + in_cluster_rank", "#FFFFFF", fontsize=8.1)
    arrow(ax, (8.35, 1.45), (7.8, 2.35), color=COLORS["blue"], lw=1.2)

    ax.text(
        6.0,
        4.05,
        "聚类信息进入训练；推理阶段硬聚类重排单独评估。",
        ha="center",
        fontsize=9,
        color="#48525C",
    )
    save(fig, "cluster_context_framework")


def stagewise_training_framework() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.set_axis_off()
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.0)

    columns = [
        ("共享编码器", "价格 / 新闻 / 因子\n共同表示", 0.45, COLORS["light_gray"]),
        ("阶段一", "分别训练\n排序 / 收益 / 方向头", 2.75, COLORS["light_blue"]),
        ("阶段二", "冻结或弱更新编码器\n校准各输出头", 5.05, COLORS["light_green"]),
        ("阶段三", "联合解冻\nfinal score 对齐", 7.35, COLORS["light_purple"]),
        ("评估", "rolling-10\nTop20 / 超额 / 回撤", 9.65, COLORS["light_orange"]),
    ]
    for i, (title, body, x, color) in enumerate(columns):
        box(ax, (x, 2.0), (1.65, 0.88), f"{title}\n{body}", color, fontsize=8.2)
        if i:
            arrow(ax, (columns[i - 1][2] + 1.65, 2.44), (x, 2.44))

    heads = [("y_score\n排序", 3.05), ("return_pred\n收益", 4.2), ("direction\n方向", 5.35)]
    for label, x in heads:
        box(ax, (x, 0.72), (1.0, 0.55), label, "#FFFFFF", fontsize=7.4)
        arrow(ax, (x + 0.5, 1.27), (x + 0.5, 2.0), lw=0.9)

    ax.text(5.9, 3.45, "分阶段训练用于检查多头监督是否能改善最终 Top20 排序", ha="center", fontsize=9.2, color=COLORS["ink"])
    save(fig, "stagewise_training_framework")


def strict_loss_score_framework() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    ax.set_axis_off()
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.1)

    left = [
        ("模型输出", "final_score\nreturn_pred\ndirection_prob", 0.55, COLORS["light_blue"]),
        ("可用标签", "open return\nVWAP return\n方向标签", 2.7, COLORS["light_green"]),
        ("严格变量", "只使用已物化字段\n不引入外部变量", 4.85, COLORS["light_gray"]),
    ]
    for i, (title, body, x, color) in enumerate(left):
        box(ax, (x, 2.35), (1.55, 0.82), f"{title}\n{body}", color, fontsize=7.9)
        if i:
            arrow(ax, (left[i - 1][2] + 1.55, 2.76), (x, 2.76))

    box(ax, (7.1, 2.55), (1.8, 0.68), "Loss 方案集\nA2 / A4 / A5 / A7", COLORS["light_purple"], fontsize=8.1)
    box(ax, (7.1, 1.55), (1.8, 0.68), "Score 方案集\nmultihead / exact", COLORS["light_orange"], fontsize=8.1)
    box(ax, (9.75, 2.0), (1.75, 0.82), "同一 holdout\n统一指标重算", "#FFFFFF", fontsize=8.1)
    arrow(ax, (6.4, 2.76), (7.1, 2.89))
    arrow(ax, (6.4, 2.76), (7.1, 1.89))
    arrow(ax, (8.9, 2.89), (9.75, 2.45))
    arrow(ax, (8.9, 1.89), (9.75, 2.45))

    ax.text(5.9, 3.55, "严格 loss/score 方案集用于验证外部设计是否能落到现有字段", ha="center", fontsize=9.2, color=COLORS["ink"])
    save(fig, "strict_loss_score_framework")


def theme_context_framework() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    ax.set_axis_off()
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.1)

    columns = [
        ("基础样本", "价格窗口\n新闻评分\n入选因子", 0.45, COLORS["light_gray"]),
        ("主题上下文", "动量强度\n资金确认\n回撤状态", 2.75, COLORS["light_green"]),
        ("上下文 token", "拼接到 MSGCA\n共享表示", 5.05, COLORS["light_blue"]),
        ("主题 score", "direct_theme\nmedium / soft", 7.35, COLORS["light_purple"]),
        ("Top20", "主题信息参与\n最终排序", 9.65, COLORS["light_orange"]),
    ]
    for i, (title, body, x, color) in enumerate(columns):
        box(ax, (x, 2.05), (1.65, 0.88), f"{title}\n{body}", color, fontsize=8.2)
        if i:
            arrow(ax, (columns[i - 1][2] + 1.65, 2.49), (x, 2.49))

    box(ax, (2.4, 0.78), (3.15, 0.68), "研究问题：粗行业字段是否不足以表达近期主题强弱", "#FFFFFF", fontsize=8.0)
    box(ax, (6.15, 0.78), (3.15, 0.68), "验证口径：holdout / rolling10 / latest10 / 超额等权", "#FFFFFF", fontsize=8.0)
    arrow(ax, (4.0, 1.46), (3.58, 2.05), lw=0.9)
    arrow(ax, (7.72, 1.46), (8.18, 2.05), lw=0.9)
    ax.text(5.9, 3.55, "主题上下文训练：把近期主题强度作为模型输入，而非推理后硬规则", ha="center", fontsize=9.2, color=COLORS["ink"])
    save(fig, "theme_context_framework")


def experiment_roadmap_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.2)

    stages = [
        ("无泄露\nbaseline", "可见性与 scaler\n边界", 0.45, 3.55, COLORS["light_blue"]),
        ("因子/模态\n消融", "识别横截面\n有效信号", 2.95, 3.55, COLORS["light_green"]),
        ("Scaler+epoch\n复训", "OOS 快照与\n续训一致性", 5.45, 3.55, COLORS["light_gray"]),
        ("Loss/TopN\n策略网格", "收益头与 TopK\n目标错位", 7.95, 3.55, COLORS["light_orange"]),
        ("Stagewise+\n窗口 loss", "对齐多头与\nrolling-10 目标", 10.45, 3.55, COLORS["light_purple"]),
        ("Strict 方案集", "按现有字段验证\nscore/loss", 0.45, 1.15, COLORS["light_gray"]),
        ("主题上下文\n训练", "近期 regime 与\n主题强度", 2.95, 1.15, COLORS["light_green"]),
        ("动态聚类\n训练", "细分主题簇\n上下文", 5.45, 1.15, COLORS["light_purple"]),
        ("策略层级\n检查", "硬聚类与\nensemble 诊断", 7.95, 1.15, COLORS["light_orange"]),
        ("预测产物\n归档", "buy list 与\n实验登记", 10.45, 1.15, COLORS["light_blue"]),
    ]
    for idx, (title, body, x, y, color) in enumerate(stages):
        box(ax, (x, y), (1.75, 0.88), f"{title}\n{body}", color, fontsize=7.9)
        if idx in (1, 2, 3, 4):
            arrow(ax, (stages[idx - 1][2] + 1.75, y + 0.44), (x, y + 0.44), lw=1.25)
        if idx == 5:
            arrow(ax, (11.32, 3.55), (1.32, 2.03), lw=1.1)
        if idx > 5:
            arrow(ax, (stages[idx - 1][2] + 1.75, y + 0.44), (x, y + 0.44), lw=1.25)

    ax.text(6.6, 4.85, "实验总览：从输入、训练目标到上下文建模", ha="center", fontsize=10, weight="bold")
    ax.text(6.6, 0.35, "路径从数据有效性、模型输入、目标对齐，推进到主题/聚类感知训练。", ha="center", fontsize=9, color="#48525C")
    save(fig, "experiment_evolution_roadmap")


def late_stage_holdout_comparison() -> None:
    pasted = read_csv_rows(EXP_DIR / "20260604_pasted_full_local" / "eval_all_txt" / "combined_competition_summary.csv")
    theme = read_csv_rows(EXP_DIR / "20260604_theme_train" / "eval" / "combined_competition_summary.csv")
    cluster = read_csv_rows(EXP_DIR / "20260605_cluster_train" / "diagnostics" / "recomputed_baseline_vs_cluster_summary.csv")
    ensemble = read_csv_rows(EXP_DIR / "20260605_cluster_train" / "eval" / "multiseed_competition_summary.csv")
    stage = read_csv_rows(EXP_DIR / "20260602_stagewise_top20" / "holdout_20260521_latest_eval" / "stage_best_holdout_summary.csv")

    rows = [
        ("Stagewise\n短窗", first_row(stage, name="latest_final"), "stage"),
        ("A2 L1\n多头", first_row(pasted, experiment_id="a2_l1_only_scratch_seed2031", eval_split="holdout", eval_variant_dir="direct_multihead"), "standard"),
        ("主题 ctx008\nmedium", first_row(theme, experiment_id="theme_ctx008_scratch_seed2031", eval_split="holdout", eval_variant_dir="direct_theme_medium"), "standard"),
        ("基线\nmedium", first_row(cluster, case="baseline_holdout_medium"), "recomputed"),
        ("聚类\nsoft", first_row(cluster, case="cluster_holdout_soft"), "recomputed"),
        ("聚类\nensemble", first_row(ensemble, experiment_id="cluster_inrank020_ensemble_seed2031_2041_2051", eval_split="holdout"), "standard"),
    ]

    labels = [r[0] for r in rows]
    period = []
    excess = []
    rolling = []
    latest = []
    score = []
    for _, row, source in rows:
        period.append(as_float(row, "period_return") * 100)
        excess.append(as_float(row, "period_excess_equal") * 100)
        if source == "stage":
            rolling.append(np.nan)
            latest.append(np.nan)
            score.append(np.nan)
        elif source == "recomputed":
            rolling.append(as_float(row, "rolling_return_mean") * 100)
            latest.append(as_float(row, "latest_window_return") * 100)
            score.append(as_float(row, "competition_score") * 100)
        else:
            rolling.append(as_float(row, "rolling_return_mean") * 100)
            latest.append(as_float(row, "latest_window_return") * 100)
            score.append(as_float(row, "competition_score") * 100)

    fig, axes = plt.subplots(2, 1, figsize=(12.4, 7.2), gridspec_kw={"height_ratios": [1.05, 1.0]})
    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, period, width=width, color=COLORS["green"], label="区间收益")
    axes[0].bar(x + width / 2, excess, width=width, color=COLORS["orange"], label="相对等权超额")
    axes[0].axhline(0, color="#333", linewidth=0.8)
    axes[0].set_title("统一 holdout 组合收益")
    axes[0].set_ylabel("%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for i, v in enumerate(period):
        axes[0].text(i - width / 2, v + (0.6 if v >= 0 else -1.2), f"{v:.1f}", ha="center", fontsize=7.5)

    width2 = 0.25
    axes[1].bar(x - width2, rolling, width=width2, color=COLORS["blue"], label="rolling-10 均值")
    axes[1].bar(x, latest, width=width2, color=COLORS["purple"], label="latest-10")
    axes[1].bar(x + width2, score, width=width2, color=COLORS["gray"], label="competition score x100")
    axes[1].axhline(0, color="#333", linewidth=0.8)
    axes[1].set_title("Rolling 窗口诊断")
    axes[1].set_ylabel("% / score x100")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8, ncol=3)
    axes[1].set_ylim(min(np.nanmin(latest), np.nanmin(rolling), np.nanmin(score)) - 2, max(np.nanmax(latest), np.nanmax(rolling), np.nanmax(score)) + 2)
    fig.tight_layout()
    save(fig, "late_stage_holdout_comparison")


def strategy_layering_diagnostics() -> None:
    layer_rows = read_csv_rows(EXP_DIR / "20260605_cluster_train" / "strategy_layering" / "strategy_layering_summary.csv")
    multi = read_csv_rows(EXP_DIR / "20260605_cluster_train" / "eval" / "multiseed_competition_summary.csv")
    base = first_row(multi, experiment_id="cluster_inrank020_scratch_seed2031", eval_split="holdout", score_variant="direct_theme_soft")
    variants = [("direct_theme_soft", as_float(base, "competition_score"), as_float(base, "period_return"))]
    keep = [
        "direct_theme_soft_cluster_boostlight",
        "direct_theme_soft_cluster_boost",
        "direct_theme_soft_cluster_booststrong",
        "direct_theme_soft_cluster12",
        "direct_theme_soft_cluster6",
    ]
    for name in keep:
        row = first_row(layer_rows, eval_split="holdout", score_variant=name)
        variants.append((name.replace("direct_theme_soft_", ""), as_float(row, "competition_score"), as_float(row, "period_return")))

    label_map = {
        "direct_theme_soft": "直接\nsoft",
        "cluster_boostlight": "轻量\nboost",
        "cluster_boost": "boost",
        "cluster_booststrong": "强\nboost",
        "cluster12": "簇12",
        "cluster6": "簇6",
    }
    labels = [label_map.get(v[0], v[0].replace("_", "\n")) for v in variants]
    score = [v[1] * 100 for v in variants]
    ret = [v[2] * 100 for v in variants]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    colors = [COLORS["green"]] + [COLORS["gray"]] * (len(labels) - 1)
    axes[0].bar(x, score, color=colors, width=0.62)
    axes[0].axhline(0, color="#333", linewidth=0.8)
    axes[0].set_title("策略层级 score")
    axes[0].set_ylabel("competition score x100")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[1].bar(x, ret, color=colors, width=0.62)
    axes[1].axhline(0, color="#333", linewidth=0.8)
    axes[1].set_title("策略层级区间收益")
    axes[1].set_ylabel("%")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    fig.tight_layout()
    save(fig, "strategy_layering_diagnostics")


def factor_ablation_figure() -> None:
    labels = [
        "仅价格",
        "仅因子\n无价格",
        "人工因子+\n价格",
        "新闻+\n价格",
        "全部入选+\n价格",
        "H48\nproto2",
    ]
    rank_ic = np.array([0.02077, 0.04496, 0.04147, 0.02266, 0.04138, 0.04245])
    total = np.array([-3.61, 6.92, 4.17, -0.30, 7.30, 11.86])
    excess = np.array([-15.08, -4.55, -7.30, -11.77, -4.16, 0.40])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(labels))
    axes[0].bar(x, rank_ic, color=COLORS["blue"], width=0.65)
    axes[0].set_title("留出期 RankIC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0, 0.052)
    axes[0].axhline(0, color="#333", linewidth=0.8)
    for i, v in enumerate(rank_ic):
        axes[0].text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=8)

    width = 0.35
    axes[1].bar(x - width / 2, total, width=width, color=COLORS["green"], label="区间收益")
    axes[1].bar(x + width / 2, excess, width=width, color=COLORS["orange"], label="相对等权超额")
    axes[1].set_title("留出期组合收益 (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].axhline(0, color="#333", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_ylim(-17, 14)
    fig.tight_layout()
    save(fig, "factor_ablation")


def structure_ablation_figure() -> None:
    groups = [
        (
            "TopK loss 权重",
            ["topk000", "topk002", "topk005"],
            np.array([0.04145, 0.04138, 0.04126]),
            np.array([7.24, 7.30, 6.41]),
            np.array([-4.23, -4.16, -5.05]),
        ),
        (
            "Prototype / gate",
            ["plain", "proto2", "no_gate"],
            np.array([0.04126, 0.04272, 0.04544]),
            np.array([6.41, 5.79, -0.75]),
            np.array([-5.05, -5.68, -12.22]),
        ),
        (
            "Hidden size",
            ["h48", "h64", "h96"],
            np.array([0.04245, 0.04272, 0.04321]),
            np.array([11.86, 5.79, -0.37]),
            np.array([0.40, -5.68, -11.83]),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), sharey=True)
    for ax, (title, labels, rank_ic, total, excess) in zip(axes, groups):
        x = np.arange(len(labels))
        width = 0.34
        ax.bar(x - width / 2, total, width=width, color=COLORS["blue"], alpha=0.88, label="区间收益")
        ax.bar(x + width / 2, excess, width=width, color=COLORS["gray"], alpha=0.55, label="相对等权超额")
        ax.axhline(0, color=COLORS["rule"], linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(-14, 14)
        ax.tick_params(axis="y", labelsize=8)

        rank_ax = ax.twinx()
        rank_ax.plot(x, rank_ic, color=COLORS["ink"], marker="o", markersize=4.2, linewidth=1.1, label="RankIC")
        rank_ax.set_ylim(0.039, 0.047)
        rank_ax.tick_params(axis="y", labelsize=7, colors=COLORS["muted"])
        rank_ax.spines["right"].set_color(COLORS["rule"])
        for i, v in enumerate(rank_ic):
            rank_ax.text(i, v + 0.00018, f"{v:.3f}", ha="center", fontsize=7, color=COLORS["ink"])

    axes[0].set_ylabel("收益 (%)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "structure_ablation")


def short_window_figure() -> None:
    labels = ["sparse", "soft", "soft\nrecent"]
    rank_ic = np.array([-0.04717, -0.04788, -0.04788])
    total = np.array([-1.81, -1.69, -1.69])
    excess = np.array([-4.04, -3.92, -3.92])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    x = np.arange(len(labels))
    axes[0].bar(x, rank_ic, color=[COLORS["gray"], COLORS["blue"], COLORS["purple"]], width=0.55)
    axes[0].axhline(0, color="#333", linewidth=0.8)
    axes[0].set_title("短窗口复训 RankIC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(-0.06, 0.01)
    for i, v in enumerate(rank_ic):
        axes[0].text(i, v - 0.004, f"{v:.3f}", ha="center", va="top", fontsize=8)

    width = 0.33
    axes[1].bar(x - width / 2, total, width=width, color=COLORS["green"], label="区间收益")
    axes[1].bar(x + width / 2, excess, width=width, color=COLORS["orange"], label="相对等权超额")
    axes[1].axhline(0, color="#333", linewidth=0.8)
    axes[1].set_title("11 日收益 (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylim(-5, 1)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "short_window_cutoff")


def selected_feature_distribution() -> None:
    data = [
        ("估值/指标", 86),
        ("rolling20", 57),
        ("rolling10", 55),
        ("rolling5", 55),
        ("rolling60", 53),
        ("rolling3", 51),
        ("资金流", 46),
        ("价格", 33),
        ("K线", 19),
        ("市场新闻", 14),
        ("收益", 13),
        ("GPT 挖掘", 73),
        ("个股新闻", 5),
    ]
    labels, values = zip(*data)
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = [COLORS["green"] if "GPT" not in labels[i] and "news" not in labels[i] else COLORS["orange"] for i in order]
    ax.barh(np.array(labels)[order], np.array(values)[order], color=colors)
    ax.set_title("reviewed 入选特征分布 (n=560)")
    ax.set_xlabel("特征数量")
    for y, v in enumerate(np.array(values)[order]):
        ax.text(v + 1.2, y, str(v), va="center", fontsize=8)
    ax.set_xlim(0, max(values) + 18)
    fig.tight_layout()
    save(fig, "selected_feature_distribution")


def main() -> None:
    workflow_figure()
    msgca_figure()
    cluster_context_framework()
    stagewise_training_framework()
    strict_loss_score_framework()
    theme_context_framework()
    late_stage_holdout_comparison()
    strategy_layering_diagnostics()
    selected_feature_distribution()
    factor_ablation_figure()
    structure_ablation_figure()
    short_window_figure()


if __name__ == "__main__":
    main()
