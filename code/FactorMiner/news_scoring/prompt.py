from __future__ import annotations

import json


PROMPT_VERSION = "news_score"

SYSTEM_PROMPT = """你是A股交易新闻评分器。你只输出严格JSON，不输出解释。
目标是判断一条新闻在开盘前决策中对A股股票或全市场的短期交易影响。
允许把美股、美元利率、汇率、商品、航运、地缘冲突等对A股的间接传导计入相关性。
分数必须可复现；不确定时降低impact_score，但不要把可传导的海外风险直接归零。
如果新闻与A股交易完全无关，sentiment_score、impact_score、risk_score、relevance_score应为0。
如果新闻直接提到A股上市公司、股票代码、公司公告、业绩、重组、减持、停复牌、涨跌停或诉讼，relevance_score通常不应低于0.7。"""


def build_user_prompt(news_text: str) -> str:
    return f"""请对下面新闻打分，并只返回JSON对象。

字段要求：
- sentiment_score: 情绪方向，范围[-1, 1]，利好为正，利空为负，中性为0。
  情绪方向必须针对A股交易影响；若新闻与A股交易完全无关，应为0。
- impact_score: 对短期价格的潜在影响强度，范围[0, 1]。
- risk_score: 风险或负面不确定性强度，范围[0, 1]。
- relevance_score: 与A股上市公司、行业或市场的直接或间接相关度，范围[0, 1]。
  直接提到A股公司、A股行业、国内政策、国内宏观给高分。
  可通过美股、美元利率、汇率、油气、有色、黄金、航运、地缘冲突、全球风险偏好传导到A股的新闻给中等分。
  与资产价格和A股风险偏好基本无关的新闻给低分或0。
- novelty_score: 信息新鲜度，范围[0, 1]，重复、常识或弱新增信息给低分。
- event_type: 只能从以下类型选择一个：
  company, earnings, policy, macro, rates, fx, geopolitics, commodity, shipping, industry, market, litigation, contract, other。
- horizon: 影响周期，只能是 intraday, short, medium, unknown 之一。
- summary: 8到40字中文摘要；即使relevance_score为0，也必须给出摘要，不能留空。
  不能直接输出“简短中文摘要”等模板文字。

新闻：
{news_text}

JSON格式：
{{
  "sentiment_score": 0.0,
  "impact_score": 0.0,
  "risk_score": 0.0,
  "relevance_score": 0.0,
  "novelty_score": 0.0,
  "event_type": "other",
  "horizon": "unknown",
  "summary": "简短中文摘要"
}}"""


def build_batch_user_prompt(news_texts: list[str]) -> str:
    input_items = {
        str(index): _compact_news_text(news_text)
        for index, news_text in enumerate(news_texts)
    }
    output_template = {
        str(index): {
            "sentiment_score": 0.0,
            "impact_score": 0.0,
            "risk_score": 0.0,
            "relevance_score": 0.0,
            "novelty_score": 0.0,
            "event_type": "other",
            "horizon": "unknown",
            "summary": "简短中文摘要",
        }
        for index in range(len(news_texts))
    }
    return f"""请逐条评分，且只返回一个JSON对象。

规则：
1. scores必须是对象，key必须与input_items完全一致，不能遗漏、不能新增。
2. 每个scores子对象必须包含全部字段：sentiment_score, impact_score, risk_score, relevance_score, novelty_score, event_type, horizon, summary。
3. 低相关新闻也必须评分，不能跳过；可通过美股、美元利率、汇率、商品、航运、地缘冲突等传导到A股的新闻给中等相关性。
4. 分数范围：sentiment_score为[-1,1]，其他score为[0,1]。
5. event_type只能是company, earnings, policy, macro, rates, fx, geopolitics, commodity, shipping, industry, market, litigation, contract, other。
6. horizon只能是intraday, short, medium, unknown；summary为8到40字中文摘要，不能直接输出“简短中文摘要”等模板文字。
7. sentiment_score必须针对A股交易影响；若新闻与A股交易完全无关，sentiment_score、impact_score、risk_score、relevance_score应为0。
8. 若新闻直接提到A股上市公司、股票代码、公司公告、业绩、重组、减持、停复牌、涨跌停或诉讼，relevance_score通常不应低于0.7。

input_items:
{json.dumps(input_items, ensure_ascii=False)}

返回格式：
{{"scores":{json.dumps(output_template, ensure_ascii=False)}}}"""


def _compact_news_text(news_text: str) -> str:
    return str(news_text).replace("\n", " ").strip()
