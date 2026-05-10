"""
全局配置：股票池参数、技术指标参数、打分权重
"""

# ── 股票池过滤 ──
MIN_MARKET_CAP = 20e8    # 最小总市值 20 亿
MAX_MARKET_CAP = 2000e8  # 最大总市值 2000 亿
EXCLUDE_ST = True        # 排除 ST / *ST

# ── 技术指标参数 ──
MA_PERIODS = [5, 10, 20, 60]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
KDJ_N = 9
BOLL_PERIOD = 20
BOLL_STD = 2

# ── 打分权重（总和 100） ──
WEIGHTS = {
    "technical":   35,   # 技术面：均线排列、MACD、量价、RSI、布林带
    "fundamental": 20,   # 基本面：PE/PB、增速（akshare 财务数据有限）
    "capital":     20,   # 资金面：主力流入、北向
    "momentum":    25,   # 动量：近期涨幅、超额收益、波动率
}

# ── 数据缓存 ──
CACHE_DIR = "stock_screener/cache"

# ── 扫描范围 ──
# 全市场扫描耗时较长，开发测试时可用 'hs300' / 'zz500' / 'all'
SCAN_MODE = "hs300"  # 默认沪深300，后续改为 'all'
