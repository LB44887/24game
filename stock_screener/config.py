"""
全局配置：股票池参数、技术指标参数、打分权重
"""

# ── 股票池过滤 ──
MIN_MARKET_CAP = 30e8    # 最小流通市值 30 亿
MAX_MARKET_CAP = 3000e8  # 最大流通市值 3000 亿
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
    "fundamental": 15,   # 基本面：市值适中、换手率合理
    "capital":     20,   # 资金面：量价配合、放量信号
    "momentum":    30,   # 动量：近期涨幅、超额收益、波动率
}

# ── 数据缓存 ──
CACHE_DIR = "stock_screener/cache"

# ── 扫描范围 ──
# 'hs300' / 'zz500' / 'all' — 全市场扫描约需 15-30 分钟
SCAN_MODE = "all"

# ── 扫描限制（全市场模式下最多扫多少只，0=不限制） ──
SCAN_LIMIT = 0  # 0 = 全部，设为 500 可快速测试
