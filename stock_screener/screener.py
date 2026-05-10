"""
多因子打分引擎：对全市场股票逐只计算特征 → 归一化打分 → 加权排序
"""

import time
import pandas as pd
import numpy as np
from .config import WEIGHTS
from .data_fetcher import fetch_stock_list, fetch_kline_batch
from .feature_engineer import compute_all_features


def run_screen(stock_list=None, limit=30, delay=0.1):
    """
    主筛选流程
    - stock_list: DataFrame (columns: code, name) 或 None（自动获取）
    - limit: 返回前 N 只
    - delay: 请求间隔（秒）
    返回: DataFrame (code, name, score, 各维度得分, 关键信号)
    """
    if stock_list is None or stock_list.empty:
        stock_list = fetch_stock_list()
    if stock_list.empty:
        return pd.DataFrame()

    codes = stock_list["code"].tolist()
    names = dict(zip(stock_list["code"], stock_list["name"]))

    print(f"  开始扫描 {len(codes)} 只股票...")
    results = []

    for i, code in enumerate(codes):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i+1}/{len(codes)}, 已完成打分: {len(results)}")

        features = _get_features_cached(code)
        if features is None:
            continue

        scores = _score_features(features)
        total = sum(scores[k] * WEIGHTS[k] / 100 for k in WEIGHTS)

        results.append({
            "code": code,
            "name": names.get(code, ""),
            "score": round(total, 1),
            "tech_score": round(scores.get("technical", 0), 1),
            "fund_score": round(scores.get("fundamental", 0), 1),
            "cap_score": round(scores.get("capital", 0), 1),
            "mom_score": round(scores.get("momentum", 0), 1),
            "signals": _summarize_signals(features),
            **features,
        })

        if delay > 0:
            time.sleep(delay)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("score", ascending=False).head(limit)
    return df.reset_index(drop=True)


def _get_features_cached(code):
    """拉取 K 线并计算特征"""
    from .data_fetcher import fetch_kline
    kline = fetch_kline(code, days=250)
    if kline.empty or len(kline) < 50:
        return None
    return compute_all_features(kline)


def _score_features(f):
    """
    将原始特征值映射为 0-100 分
    每个维度返回一个综合分
    """
    scores = {}

    # ── 技术面 (technical) ─────────────────
    tech = 0
    tech += f.get("ma_alignment", 0) * 0.25        # 均线排列 0-100
    # MACD 金叉或看涨
    tech += f.get("macd_golden_cross", 0) * 0.20
    tech += f.get("macd_bullish", 0) * 0.15
    # RSI: 40-70 区间最优 (非超卖超买)
    rsi = f.get("rsi", 50)
    tech += (60 if 40 <= rsi <= 70 else (30 if 30 <= rsi <= 80 else 10)) * 0.10
    # KDJ 金叉
    tech += f.get("kdj_golden_cross", 0) * 0.10
    # 布林带位置: 20-80 区间最佳
    bb = f.get("bb_position", 50)
    tech += (70 if 20 <= bb <= 80 else (40 if 10 <= bb <= 90 else 10)) * 0.05
    # 量价配合
    tech += f.get("vol_price_match", 50) * 0.10
    tech += f.get("vol_breakout", 0) * 0.05
    scores["technical"] = min(100, tech)

    # ── 基本面 (fundamental) — 有限数据 ───
    # 从实时行情补充 PE/PB 信息（如果可用）
    fund = 50  # 基础分，数据有限
    # 可以通过扩展 data_fetcher 补充 PE/PB 后细化
    scores["fundamental"] = fund

    # ── 资金面 (capital) — 有限数据 ───────
    # 量价关系可部分反映资金面
    cap = 50
    cap += (f.get("vol_breakout", 0) - 50) * 0.3  # 放量说明资金关注
    cap += (f.get("vol_price_match", 50) - 50) * 0.3
    scores["capital"] = min(100, max(0, cap))

    # ── 动量 (momentum) ───────────────────
    mom = 0
    mom += f.get("momentum_score", 50) * 0.30
    mom += f.get("vol_score", 50) * 0.20          # 低波动 = 高分
    mom += f.get("near_20d_high", 50) * 0.15       # 接近新高
    mom += f.get("bullish_ratio", 50) * 0.15       # 阳线比例
    # 连涨加分
    streak = f.get("up_streak", 0)
    mom += (min(streak, 5) / 5 * 100) * 0.10
    # 价格在 MA20 上方
    p_ma = f.get("price_vs_ma20", 0)
    mom += (80 if -2 <= p_ma <= 10 else (50 if -5 <= p_ma <= 20 else 20)) * 0.10
    scores["momentum"] = min(100, mom)

    return scores


def _summarize_signals(f):
    """生成简短的信号摘要文本"""
    signals = []
    if f.get("ma_alignment", 0) >= 75:
        signals.append("多头排列")
    if f.get("macd_golden_cross", 0) >= 100:
        signals.append("MACD金叉")
    elif f.get("macd_bullish", 0) >= 100:
        signals.append("MACD看涨")
    if f.get("kdj_golden_cross", 0) >= 100:
        signals.append("KDJ金叉")
    if f.get("vol_breakout", 0) >= 100:
        signals.append("放量突破")
    if f.get("vol_price_match", 0) >= 100:
        signals.append("放量上涨")
    if f.get("up_streak", 0) >= 3:
        signals.append(f"连涨{f['up_streak']}日")
    if f.get("ret_20d", -99) > 5 and f.get("ret_20d", 0) < 25:
        signals.append("趋势上行")
    if f.get("near_20d_high", 0) >= 100:
        signals.append("接近新高")
    return ", ".join(signals) if signals else "关注"
