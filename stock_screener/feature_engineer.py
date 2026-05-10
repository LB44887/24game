"""
特征计算：基于 K 线数据计算技术指标 + 动量因子
纯 pandas 实现，不依赖 TA-Lib
"""

import pandas as pd
import numpy as np
from .config import MA_PERIODS, MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD, BOLL_PERIOD, BOLL_STD


def compute_all_features(kline_df):
    """
    输入: 单只股票的 K 线 DataFrame (columns: date,open,high,low,close,volume)
    输出: dict，包含所有特征因子
    """
    if kline_df.empty or len(kline_df) < 60:
        return None

    df = kline_df.copy()
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    features = {}

    # ── 1. 均线 ─────────────────────────────
    mas = {}
    for p in MA_PERIODS:
        ma = _sma(close, p)
        mas[p] = ma
        features[f"ma{p}"] = ma[-1]

    # 均线多头排列得分: 5>10>20>60 得满分
    ma_values = [mas[p][-1] for p in sorted(MA_PERIODS)]
    align_score = 0
    for i in range(len(ma_values) - 1):
        if ma_values[i] > ma_values[i+1]:
            align_score += 25
    features["ma_alignment"] = align_score

    # 股价相对 MA20 位置 (%)
    ma20 = mas[20][-1]
    features["price_vs_ma20"] = (close[-1] / ma20 - 1) * 100 if ma20 > 0 else 0

    # ── 2. MACD ────────────────────────────
    ema_fast = _ema(close, MACD_FAST)
    ema_slow = _ema(close, MACD_SLOW)
    dif = ema_fast - ema_slow
    dea = _ema(dif, MACD_SIGNAL)
    macd_hist = 2 * (dif - dea)

    features["macd_dif"] = dif[-1]
    features["macd_dea"] = dea[-1]
    features["macd_hist"] = macd_hist[-1]

    # 金叉检测 (近5日内DIF上穿DEA)
    golden_cross = False
    for i in range(max(0, len(dif)-5), len(dif)):
        if dif[i] > dea[i] and dif[i-1] <= dea[i-1]:
            golden_cross = True
            break
    features["macd_golden_cross"] = 100 if golden_cross else 0

    # MACD 趋势 (DIF > DEA 且 DIF > 0)
    features["macd_bullish"] = 100 if (dif[-1] > dea[-1] and dif[-1] > 0) else (50 if dif[-1] > dea[-1] else 0)

    # ── 3. RSI ─────────────────────────────
    rsi = _rsi(close, RSI_PERIOD)
    features["rsi"] = rsi[-1] if not np.isnan(rsi[-1]) else 50

    # ── 4. KDJ ─────────────────────────────
    k, d, j = _kdj(high, low, close, n=9)
    features["kdj_k"] = k[-1]
    features["kdj_d"] = d[-1]
    features["kdj_j"] = j[-1]
    # KDJ 金叉
    kdj_golden = False
    for i in range(max(0, len(k)-3), len(k)):
        if k[i] > d[i] and k[i-1] <= d[i-1]:
            kdj_golden = True
            break
    features["kdj_golden_cross"] = 100 if kdj_golden else 0

    # ── 5. 布林带 ──────────────────────────
    bb_mid = _sma(close, BOLL_PERIOD)
    bb_std = _rolling_std(close, BOLL_PERIOD)
    bb_upper = bb_mid + BOLL_STD * bb_std
    bb_lower = bb_mid - BOLL_STD * bb_std
    bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_mid[-1] * 100 if bb_mid[-1] > 0 else 0
    bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
    features["bb_position"] = bb_position * 100  # 0-100
    features["bb_width"] = bb_width

    # ── 6. 量价 ────────────────────────────
    vol_ma20 = _sma(volume, 20)
    vol_ratio = volume[-1] / vol_ma20[-1] if vol_ma20[-1] > 0 else 1
    features["vol_ratio"] = vol_ratio  # >1.5 = 放量
    features["vol_breakout"] = 100 if vol_ratio > 1.5 else (50 if vol_ratio > 1.2 else 0)

    # 量价配合: 放量上涨 = 高分
    if close[-1] > close[-2] and vol_ratio > 1.2:
        features["vol_price_match"] = 100
    elif close[-1] < close[-2] and vol_ratio > 1.2:
        features["vol_price_match"] = 0  # 放量下跌 = 差
    else:
        features["vol_price_match"] = 50

    # ── 7. 动量 ────────────────────────────
    features["ret_5d"] = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
    features["ret_10d"] = (close[-1] / close[-11] - 1) * 100 if len(close) >= 11 else 0
    features["ret_20d"] = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0

    # 超额收益: 20日涨幅不宜过大(追高风险)，3-15%区间最优
    ret_20 = features["ret_20d"]
    if 3 <= ret_20 <= 15:
        features["momentum_score"] = 100
    elif 0 <= ret_20 <= 25:
        features["momentum_score"] = 60
    elif ret_20 < 0:
        features["momentum_score"] = 20  # 下跌趋势扣分
    else:
        features["momentum_score"] = 30  # 涨幅过大

    # ── 8. 波动率 ──────────────────────────
    daily_ret = np.diff(close) / close[:-1]
    vol_20d = np.std(daily_ret[-20:]) * 100 if len(daily_ret) >= 20 else 0
    features["volatility_20d"] = vol_20d
    # 低波动上涨更好: 波动率 < 2% 满分
    if vol_20d < 1.5:
        features["vol_score"] = 100
    elif vol_20d < 2.5:
        features["vol_score"] = 70
    elif vol_20d < 4:
        features["vol_score"] = 40
    else:
        features["vol_score"] = 10

    # ── 9. 近期趋势强度 ────────────────────
    # 计算过去20日有多少天收盘 > 开盘 (阳线比例)
    bullish_days = np.sum((close[-20:] - df["open"].values[-20:].astype(float)) > 0)
    features["bullish_ratio"] = bullish_days / 20 * 100

    # 连涨天数
    up_streak = 0
    for i in range(len(close)-1, 0, -1):
        if close[i] > close[i-1]:
            up_streak += 1
        else:
            break
    features["up_streak"] = up_streak

    # ── 10. 创新高能力 ─────────────────────
    high_20d = np.max(high[-20:])
    features["near_20d_high"] = 100 if close[-1] >= high_20d * 0.97 else (50 if close[-1] >= high_20d * 0.9 else 0)

    return features


# ── 底层指标计算函数 ───────────────────────────────

def _sma(series, period):
    """简单移动平均"""
    result = np.full_like(series, np.nan, dtype=float)
    for i in range(period-1, len(series)):
        result[i] = np.mean(series[i-period+1:i+1])
    return result


def _ema(series, period):
    """指数移动平均，自动跳过 NaN"""
    result = np.full_like(series, np.nan, dtype=float)

    # 找第一个非 NaN 的位置
    valid_start = 0
    while valid_start < len(series) and np.isnan(series[valid_start]):
        result[valid_start] = np.nan
        valid_start += 1
    if valid_start + period > len(series):
        return result

    start_idx = valid_start + period - 1
    # 初始 SMA 用非 NaN 值
    seg = series[valid_start:start_idx+1]
    result[start_idx] = np.mean(seg[~np.isnan(seg)]) if np.any(~np.isnan(seg)) else series[start_idx]

    multiplier = 2 / (period + 1)
    for i in range(start_idx + 1, len(series)):
        if np.isnan(series[i]):
            result[i] = result[i-1]
        else:
            result[i] = (series[i] - result[i-1]) * multiplier + result[i-1]
    return result


def _rsi(close, period=14):
    """RSI 指标"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)

    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period

    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _kdj(high, low, close, n=9):
    """KDJ 指标"""
    k = np.full_like(close, np.nan, dtype=float)
    d = np.full_like(close, np.nan, dtype=float)
    j = np.full_like(close, np.nan, dtype=float)

    for i in range(n-1, len(close)):
        hh = np.max(high[i-n+1:i+1])
        ll = np.min(low[i-n+1:i+1])
        rsv = (close[i] - ll) / (hh - ll) * 100 if hh != ll else 50

        if i == n-1:
            k[i] = 50
            d[i] = 50
        else:
            k[i] = 2/3 * k[i-1] + 1/3 * rsv
            d[i] = 2/3 * d[i-1] + 1/3 * k[i]
        j[i] = 3 * k[i] - 2 * d[i]

    return k, d, j


def _rolling_std(series, period):
    """滚动标准差"""
    result = np.full_like(series, np.nan, dtype=float)
    for i in range(period-1, len(series)):
        result[i] = np.std(series[i-period+1:i+1])
    return result
