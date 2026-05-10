"""
回测引擎：在历史时间段内运行策略，评估表现
策略：定期运行筛选器，等权买入 Top-N，持有一段时间后卖出
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .data_fetcher import fetch_kline


def backtest(codes, start_date, end_date, top_n=10, hold_days=5, rebalance_freq="weekly"):
    """
    简化回测
    - codes: 股票池（需提前拉好历史数据）
    - start_date / end_date: 回测区间 (str, YYYYMMDD)
    - top_n: 每期选前几只
    - hold_days: 持仓天数
    - rebalance_freq: 调仓频率 'weekly' / 'daily' / 'monthly'
    """
    trades = []
    equity_curve = []

    # 生成调仓日期
    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # 工作日
    if rebalance_freq == "weekly":
        rebalance_dates = dates[dates.weekday == 2]  # 周三调仓
    elif rebalance_freq == "monthly":
        rebalance_dates = dates[dates.is_month_start]
    else:
        rebalance_dates = dates[::hold_days]

    cash = 1_000_000  # 初始资金 100 万
    holdings = {}     # {code: shares}
    total_value = cash
    last_value = cash
    daily_values = []

    for i, today in enumerate(rebalance_dates):
        today_str = today.strftime("%Y%m%d")

        # 卖出上期持仓
        for code, shares in list(holdings.items()):
            price = _get_price_on_date(code, today_str)
            if price:
                cash += shares * price
        holdings = {}
        if cash <= 0:
            break

        # 运行筛选（用当天之前的250天数据）
        rankings = []
        for code in codes:
            kline = fetch_kline(code, days=250)
            if kline.empty or len(kline) < 50:
                continue
            # 只用当天之前的数据
            kline = kline[kline["date"] <= today]
            if len(kline) < 50:
                continue

            from .feature_engineer import compute_all_features
            feat = compute_all_features(kline)
            if feat is None:
                continue

            from .screener import _score_features
            scores = _score_features(feat)
            from .config import WEIGHTS
            total = sum(scores[k] * WEIGHTS[k] / 100 for k in WEIGHTS)
            rankings.append({"code": code, "score": total})

        # 选 Top-N
        rankings.sort(key=lambda x: x["score"], reverse=True)
        picks = rankings[:top_n]

        if not picks:
            continue

        # 等权买入
        per_stock = cash / len(picks)
        for p in picks:
            price = _get_price_on_date(p["code"], today_str)
            if price and price > 0:
                shares = int(per_stock / price / 100) * 100  # 整手
                if shares > 0:
                    holdings[p["code"]] = shares
                    cash -= shares * price
                    trades.append({
                        "date": today_str, "code": p["code"],
                        "action": "buy", "price": price,
                        "shares": shares, "score": p["score"],
                    })

        # 记录净值
        portfolio_value = cash + sum(
            h * (_get_price_on_date(c, today_str) or 0)
            for c, h in holdings.items()
        )
        daily_values.append({"date": today, "value": portfolio_value})

    # 计算指标
    if not daily_values:
        return {"error": "无有效交易日"}, pd.DataFrame()

    values = [d["value"] for d in daily_values]
    returns = np.diff(values) / values[:-1]

    win_rate = np.sum(np.array(returns) > 0) / len(returns) * 100 if len(returns) > 0 else 0
    total_return = (values[-1] / values[0] - 1) * 100 if values[0] > 0 else 0

    # 夏普比率（简化：无风险利率=0.02）
    if len(returns) > 1:
        excess = np.array(returns) - 0.02 / 252
        sharpe = np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252)
    else:
        sharpe = 0

    # 最大回撤
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    max_dd = abs(max_dd)

    report = {
        "total_return_pct": round(total_return, 2),
        "win_rate_pct": round(win_rate, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": len(trades),
        "period": f"{start_date} ~ {end_date}",
        "final_value": round(values[-1], 2),
    }

    curve_df = pd.DataFrame(daily_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return report, curve_df, trades_df


def _get_price_on_date(code, date_str):
    """获取指定日期的收盘价"""
    kline = fetch_kline(code, days=365)
    if kline.empty:
        return None
    target = pd.Timestamp(date_str)
    match = kline[kline["date"] <= target]
    if match.empty:
        return None
    return float(match.iloc[-1]["close"])
