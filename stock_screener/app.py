"""
Streamlit 股票筛选器界面
启动: streamlit run stock_screener/app.py
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_screener.config import WEIGHTS, SCAN_MODE
from stock_screener.data_fetcher import fetch_stock_list, fetch_kline, fetch_spot_data
from stock_screener.screener import run_screen
from stock_screener.feature_engineer import compute_all_features

st.set_page_config(
    page_title="AI 选股系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 样式 ──
st.markdown("""
<style>
.main-header { font-size: 2rem; font-weight: 900; color: #FFD700;
    background: linear-gradient(135deg, #1a3a1a, #0a3d15); padding: 1rem 2rem; border-radius: 12px; text-align: center; margin-bottom: 1rem; }
.stock-up { color: #ef5350; font-weight: bold; }
.stock-down { color: #26a69a; font-weight: bold; }
.metric-card { background: #f5f5f5; padding: 1rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── 侧边栏 ──
with st.sidebar:
    st.title("📊 控制面板")
    page = st.radio("导航", ["🏠 每日选股", "🔍 单股分析", "📋 策略回测"])

    st.divider()
    scan = st.selectbox("扫描范围", ["hs300", "zz500", "top50"], index=0)
    if scan != SCAN_MODE:
        st.session_state["scan_mode"] = scan

    st.divider()
    st.caption("⚠️ 风险提示")
    st.caption("本工具仅供学习研究，不构成投资建议。股市有风险，投资需谨慎。")


# ── 每日选股 ──
if page == "🏠 每日选股":
    st.markdown('<div class="main-header">📈 AI 选股系统 — 每日精选</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_n = st.number_input("返回数量", 5, 50, 20)
    with col2:
        st.metric("扫描模式", scan.upper())
    with col3:
        from stock_screener.data_fetcher import SCAN_MODE as cfg_scan
        st.metric("股票池", f"{cfg_scan.upper()}")

    if st.button("🚀 开始筛选", type="primary", use_container_width=True):
        with st.spinner("正在扫描全市场，计算技术指标..."):
            stock_list = fetch_stock_list(force_refresh=True)
            st.info(f"股票池: {len(stock_list)} 只")

            df = run_screen(stock_list=stock_list, limit=top_n, delay=0.05)

        if df.empty:
            st.error("未找到符合条件的股票")
        else:
            st.success(f"筛选完成: {len(df)} 只")

            # 主要结果表
            display_cols = ["code", "name", "score", "tech_score", "mom_score", "signals"]
            display_df = df[[c for c in display_cols if c in df.columns]].copy()
            display_df.columns = ["代码", "名称", "总分", "技术面", "动量", "信号"]

            # 颜色高亮
            def highlight_scores(val):
                if isinstance(val, (int, float)):
                    if val >= 70: return "background-color: #c8e6c9; font-weight: bold"
                    if val >= 50: return "background-color: #fff9c4"
                return ""

            st.dataframe(
                display_df.style.applymap(highlight_scores, subset=["总分", "技术面", "动量"]),
                use_container_width=True, hide_index=True, height=600,
            )

            # 下载
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下载 CSV", csv, "stock_picks.csv", "text/csv")


# ── 单股分析 ──
elif page == "🔍 单股分析":
    st.markdown('<div class="main-header">🔍 单股深度分析</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        code = st.text_input("股票代码", "000001", max_chars=6)
        if st.button("查询", type="primary"):
            st.session_state["analyze_code"] = code

    code = st.session_state.get("analyze_code", "000001")

    with st.spinner(f"加载 {code} 数据..."):
        kline = fetch_kline(code, days=250)
        spot = fetch_spot_data()
        stock_info = spot[spot["code"] == code] if not spot.empty else pd.DataFrame()
        name = stock_info.iloc[0]["name"] if not stock_info.empty else code

    if kline.empty:
        st.error(f"未找到 {code} 的数据")
    else:
        # 股票信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("名称", name)
        with col2:
            price = float(kline.iloc[-1]["close"])
            prev = float(kline.iloc[-2]["close"]) if len(kline) > 1 else price
            chg = (price / prev - 1) * 100
            st.metric("最新价", f"{price:.2f}", f"{chg:+.2f}%")
        with col3:
            vol = float(kline.iloc[-1]["volume"])
            avg_vol = float(kline["volume"].tail(20).mean())
            st.metric("成交量", f"{vol/10000:.0f}万", f"{(vol/avg_vol-1)*100:+.0f}% vs 均量")
        with col4:
            st.metric("数据条数", len(kline))

        # K 线图 + 指标
        st.divider()
        st.subheader("📊 K 线图 + 技术指标")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("K线 + 均线 + 布林带", "成交量", "MACD"),
        )

        df = kline.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 计算均线
        for p in [5, 10, 20, 60]:
            df[f"ma{p}"] = df["close"].rolling(p).mean()

        # 布林带
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_up"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lo"] = df["bb_mid"] - 2 * df["bb_std"]

        # K 线
        colors = ["#ef5350" if df.iloc[i]["close"] >= df.iloc[i]["open"] else "#26a69a" for i in range(len(df))]
        fig.add_trace(go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="K线", showlegend=False,
        ), row=1, col=1)

        for p, color in [(5, "blue"), (10, "orange"), (20, "purple"), (60, "red")]:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[f"ma{p}"], mode="lines",
                name=f"MA{p}", line=dict(width=1, color=color),
            ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_up"], mode="lines",
            name="BB上", line=dict(width=1, color="gray", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lo"], mode="lines",
            name="BB下", line=dict(width=1, color="gray", dash="dash")), row=1, col=1)

        # 成交量
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"], name="成交量",
            marker_color=colors, showlegend=False,
        ), row=2, col=1)

        # MACD (快速计算)
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9).mean()
        hist = 2 * (dif - dea)

        fig.add_trace(go.Scatter(x=df["date"], y=dif, mode="lines", name="DIF", line=dict(color="blue")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=dea, mode="lines", name="DEA", line=dict(color="orange")), row=3, col=1)
        fig.add_trace(go.Bar(x=df["date"], y=hist, name="HIST",
            marker_color=["#ef5350" if h >= 0 else "#26a69a" for h in hist], showlegend=False), row=3, col=1)

        fig.update_layout(height=700, template="plotly_white", hovermode="x unified")
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 特征详情
        st.divider()
        st.subheader("📋 技术特征")
        feat = compute_all_features(kline)
        if feat:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MA 排列得分", f"{feat['ma_alignment']}/100")
                st.metric("RSI", f"{feat['rsi']:.1f}")
                st.metric("股价 vs MA20", f"{feat['price_vs_ma20']:+.1f}%")
                st.metric("波动率(20日)", f"{feat['volatility_20d']:.2f}%")
            with col2:
                st.metric("MACD 金叉", "是" if feat["macd_golden_cross"] > 50 else "否")
                st.metric("MACD 看涨", "是" if feat["macd_bullish"] > 50 else "否")
                st.metric("放量突破", "是" if feat["vol_breakout"] > 50 else "否")
                st.metric("量价配合", f"{feat['vol_price_match']}/100")
            with col3:
                st.metric("5日收益", f"{feat['ret_5d']:+.2f}%")
                st.metric("20日收益", f"{feat['ret_20d']:+.2f}%")
                st.metric("阳线比例", f"{feat['bullish_ratio']:.0f}%")
                st.metric("连涨天数", f"{feat['up_streak']} 天")


# ── 策略回测 ──
elif page == "📋 策略回测":
    st.markdown('<div class="main-header">📋 策略回测</div>', unsafe_allow_html=True)

    st.warning("⚠️ 回测需要较多计算资源，建议小范围测试")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.text_input("开始日期", "20250101")
    with col2:
        end_date = st.text_input("结束日期", "20260430")
    with col3:
        top_n = st.number_input("Top-N", 3, 30, 10)
    with col4:
        hold_days = st.number_input("持仓天数", 1, 30, 5)

    if st.button("▶ 运行回测", type="primary"):
        stock_list = fetch_stock_list()
        codes = stock_list["code"].head(50).tolist()  # 测试用50只

        with st.spinner(f"回测 {start_date} ~ {end_date}，共 {len(codes)} 只股票..."):
            from stock_screener.backtester import backtest
            result = backtest(codes, start_date, end_date, top_n=top_n, hold_days=hold_days)
            if len(result) == 3:
                report, curve_df, trades_df = result
            else:
                report, curve_df = result
                trades_df = pd.DataFrame()

        if "error" in report:
            st.error(report["error"])
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总收益", f"{report['total_return_pct']:+.2f}%")
            col2.metric("胜率", f"{report['win_rate_pct']:.1f}%")
            col3.metric("夏普比率", f"{report['sharpe']:.2f}")
            col4.metric("最大回撤", f"{report['max_drawdown_pct']:.2f}%")

            if not curve_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve_df["date"], y=curve_df["value"],
                    mode="lines", fill="tozeroy", name="净值",
                    line=dict(color="#4CAF50"),
                ))
                fig.update_layout(title="净值曲线", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            st.metric("交易笔数", report["total_trades"])
            if not trades_df.empty:
                st.dataframe(trades_df.head(50), use_container_width=True)


# ── 底部 ──
st.divider()
st.caption("数据来源: akshare (Sina/Tencent) | 技术指标: 纯 pandas 计算 | 仅供学习研究，不构成投资建议")
