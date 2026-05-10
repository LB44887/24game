"""
数据采集：akshare (Sina + Tencent 源) 获取 A 股行情
East Money 接口在此网络不可用，全部使用 Sina/Tencent 备用源
"""

import os
import time
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from .config import CACHE_DIR, SCAN_MODE

os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.csv")


def _read_cache(name, max_age_hours=6):
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    age = (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds()
    if age > max_age_hours * 3600:
        return None
    return pd.read_csv(path, dtype={"code": str})


def _write_cache(name, df):
    df.to_csv(_cache_path(name), index=False)


# ── 1. 股票列表 ──────────────────────────────────────

def fetch_stock_list(force_refresh=False):
    """获取全 A 股列表（代码 + 名称）"""
    if not force_refresh:
        cached = _read_cache("stock_list", max_age_hours=12)
        if cached is not None and len(cached) > 1000:
            return cached

    try:
        df = ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "code", "name": "name"})
        df["code"] = df["code"].astype(str).str.zfill(6)

        # 排除 ST / 退市
        mask = ~df["name"].str.contains("ST|退", na=False)
        df = df[mask]

        # 排除 B 股、北交所 (8/9开头=沪市主板, 0/3=深市, 4/8=创业板, 688=科创板)
        # 保留主流板块
        df = df[df["code"].str.match(r"^(0[036]|3[0]\d|4\d{5}|6[0-8]\d{4}|9\d{5})")]

        # 扫描模式
        if SCAN_MODE == "hs300":
            try:
                hs300 = ak.index_stock_cons_csindex(symbol="000300")
                codes = set(hs300["成分券代码"].astype(str).str.zfill(6))
                df = df[df["code"].isin(codes)]
            except Exception:
                pass

        _write_cache("stock_list", df)
        print(f"  股票列表: {len(df)} 只 (mode={SCAN_MODE})")
        return df
    except Exception as e:
        print(f"  股票列表获取失败: {e}")
        # 尝试读旧缓存
        old = _read_cache("stock_list", max_age_hours=9999)
        return old if old is not None else pd.DataFrame()


# ── 2. 实时行情（Sina 源） ───────────────────────────

def fetch_spot_data(force_refresh=False):
    """获取全市场实时行情（价格/涨跌幅/成交量等）"""
    if not force_refresh:
        cached = _read_cache("spot_data", max_age_hours=1)
        if cached is not None and len(cached) > 500:
            return cached

    try:
        df = ak.stock_zh_a_spot()
        rename_map = {
            "代码": "code", "名称": "name", "最新价": "price",
            "涨跌幅": "pct_chg", "涨跌额": "change",
            "成交量": "volume", "成交额": "amount",
            "今开": "open", "最高": "high", "最低": "low", "昨收": "prev_close",
            "换手率": "turnover",
        }
        df = df.rename(columns=rename_map)
        df["code"] = df["code"].astype(str).str[-6:]  # 去掉市场前缀

        keep = [c for c in rename_map.values() if c in df.columns]
        df = df[keep]
        _write_cache("spot_data", df)
        return df
    except Exception as e:
        print(f"  实时行情失败: {e}")
        return pd.DataFrame()


# ── 3. 日 K 线（Sina 源） ───────────────────────────

def fetch_kline(code, days=250):
    """获取单只股票日K线（新浪源，前复权，含成交量）"""
    cache_name = f"kline_{code}"
    cached = _read_cache(cache_name, max_age_hours=4)
    if cached is not None and len(cached) >= min(days, 20):
        return cached

    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days + 120)).strftime("%Y%m%d")

        market = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = market + code

        # Sina 接口，列名已是英文: date,open,high,low,close,volume
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]
        df = df.sort_values("date").tail(days)
        df = df.reset_index(drop=True)

        _write_cache(cache_name, df)
        return df
    except Exception:
        return pd.DataFrame()


# ── 4. 批量拉取 K 线（带延时防封） ───────────────────

def fetch_kline_batch(codes, days=250, delay=0.15):
    """批量拉取多只股票的K线，返回 dict: {code: DataFrame}"""
    results = {}
    for i, code in enumerate(codes):
        kline = fetch_kline(code, days)
        if not kline.empty and len(kline) >= 30:
            results[code] = kline
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(codes)}")
        if delay > 0:
            time.sleep(delay)
    return results
