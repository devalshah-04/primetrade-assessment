# ── Primetrade.ai Assessment — Streamlit Dashboard ───────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Primetrade — Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

# ── Load and process data (cached so it only runs once) ───────────────────────
@st.cache_data
def load_and_process():
    # Load datasets
    fg = pd.read_csv('data/fear_greed_index.csv')
    ht = pd.read_csv('data/historical_data.csv')

    # Clean and merge
    fg['date'] = pd.to_datetime(fg['date']).dt.date
    ht['date'] = pd.to_datetime(
        ht['Timestamp IST'], format='%d-%m-%Y %H:%M'
    ).dt.date

    df = pd.merge(
        ht, fg[['date','value','classification']],
        on='date', how='inner'
    )
    df = df.rename(columns={
        'value'          : 'fg_score',
        'classification' : 'sentiment',
        'Closed PnL'     : 'closed_pnl',
        'Size USD'       : 'size_usd',
        'Start Position' : 'start_position',
        'Execution Price': 'exec_price',
        'Side'           : 'side',
        'Direction'      : 'direction',
        'Account'        : 'account',
        'Coin'           : 'coin',
        'Fee'            : 'fee'
    })

    sentiment_order = [
        'Extreme Fear','Fear','Neutral','Greed','Extreme Greed'
    ]

    # Closed trades only
    trades_closed = df[df['closed_pnl'] != 0].copy()
    trades_closed['is_win'] = trades_closed['closed_pnl'] > 0

    # Trader profiles
    trader_profile = trades_closed.groupby('account').agg(
        total_pnl      = ('closed_pnl', 'sum'),
        avg_pnl        = ('closed_pnl', 'mean'),
        win_rate       = ('is_win',     'mean'),
        total_trades   = ('closed_pnl', 'count'),
        avg_trade_size = ('size_usd',   'mean'),
    ).reset_index()

    trader_profile = trader_profile.sort_values(
        'total_pnl', ascending=False
    ).reset_index(drop=True)
    trader_profile['trader_id'] = [
        'T' + str(i+1).zfill(2) for i in range(len(trader_profile))
    ]

    id_map = dict(zip(trader_profile['account'], trader_profile['trader_id']))
    df['trader_id']            = df['account'].map(id_map)
    trades_closed['trader_id'] = trades_closed['account'].map(id_map)

    # Sentiment performance
    sentiment_perf = trades_closed.groupby('sentiment').agg(
        avg_pnl      = ('closed_pnl', 'mean'),
        win_rate     = ('is_win',     'mean'),
        total_trades = ('closed_pnl', 'count'),
        avg_size     = ('size_usd',   'mean'),
    ).reset_index()
    sentiment_perf['sentiment'] = pd.Categorical(
        sentiment_perf['sentiment'],
        categories=sentiment_order, ordered=True
    )
    sentiment_perf = sentiment_perf.sort_values('sentiment')

    # Long/short bias
    directional = df[df['direction'].isin(['Open Long','Open Short'])].copy()
    bias = directional.groupby(
        ['sentiment','direction']
    ).size().unstack(fill_value=0).reset_index()
    bias['total']     = bias['Open Long'] + bias['Open Short']
    bias['pct_long']  = bias['Open Long']  / bias['total'] * 100
    bias['pct_short'] = bias['Open Short'] / bias['total'] * 100
    bias['sentiment'] = pd.Categorical(
        bias['sentiment'], categories=sentiment_order, ordered=True
    )
    bias = bias.sort_values('sentiment')

    # Clustering
    fear_perf = trades_closed[
        trades_closed['sentiment'].isin(['Fear','Extreme Fear'])
    ].groupby('account').agg(
        fear_avg_pnl  = ('closed_pnl', 'mean'),
        fear_win_rate = ('is_win',     'mean'),
    ).reset_index()

    greed_perf = trades_closed[
        trades_closed['sentiment'].isin(['Greed','Extreme Greed'])
    ].groupby('account').agg(
        greed_avg_pnl  = ('closed_pnl', 'mean'),
        greed_win_rate = ('is_win',     'mean'),
    ).reset_index()

    cluster_df = trader_profile[[
        'account','trader_id','total_pnl','win_rate',
        'total_trades','avg_trade_size'
    ]].copy()
    cluster_df = cluster_df\
        .merge(fear_perf,  on='account', how='left')\
        .merge(greed_perf, on='account', how='left')\
        .fillna(0)

    cluster_features = [
        'total_pnl','win_rate','total_trades','avg_trade_size',
        'fear_avg_pnl','greed_avg_pnl','fear_win_rate','greed_win_rate'
    ]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df[cluster_features])
    km       = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_df['cluster'] = km.fit_predict(X_scaled)

    archetype_map = {
        cluster_df.groupby('cluster')['win_rate'].mean().idxmax()       : '🎯 Precision Trader',
        cluster_df.groupby('cluster')['total_pnl'].mean().idxmax()      : '🐋 Whale Trader',
        cluster_df.groupby('cluster')['total_trades'].mean().idxmax()   : '⚡ High Frequency',
    }
    for i in range(4):
        if i not in archetype_map:
            archetype_map[i] = '📈 Developing Trader'
    cluster_df['archetype'] = cluster_df['cluster'].map(archetype_map)

    trader_profile = trader_profile.merge(
        cluster_df[['account','archetype']], on='account', how='left'
    )
    trades_closed = trades_closed.merge(
        cluster_df[['account','archetype']], on='account', how='left'
    )

    return df, trades_closed, trader_profile, sentiment_perf, bias, cluster_df, sentiment_order

# ── Load data ─────────────────────────────────────────────────────────────────
df, trades_closed, trader_profile, sentiment_perf, bias, cluster_df, sentiment_order = load_and_process()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📈 Primetrade.ai")
st.sidebar.caption("Trader Performance vs Market Sentiment")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Sentiment Analysis",
    "👤 Trader Explorer",
    "🎯 Strategy Signal"
])

colors = ['#d32f2f','#ef5350','#ffb300','#66bb6a','#2e7d32']

# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("📈 Trader Performance vs Market Sentiment")
    st.caption("Hyperliquid × Bitcoin Fear/Greed Index | May 2023 – May 2025")
    st.divider()

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Traders",    "32")
    c2.metric("Total Trades",     "211,218")
    c3.metric("Closed Trades",    "104,402")
    c4.metric("Days Analyzed",    "479")
    c5.metric("Model Accuracy",   "64.7%")
    st.divider()

    # Key insights
    st.subheader("🔍 Key Findings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
**Contrarians Earn 3.1x More**

4 traders using a contrarian strategy
earn **$803K avg** vs $259K for
trend followers.
        """)
    with col2:
        st.success("""
**Fear = Maximum Opportunity**

Fear days drive the highest trade
volume (29,808) and second-highest
avg PnL ($112.6/trade).
        """)
    with col3:
        st.warning("""
**Sentiment is a Weak Predictor**

Market mood ranks LAST in feature
importance. Trader momentum predicts
next-day profitability far more.
        """)

    st.divider()
    st.subheader("👤 Trader Archetypes")
    arch_stats = cluster_df.groupby('archetype').agg(
        Count        = ('trader_id',    'count'),
        Avg_PnL      = ('total_pnl',    'mean'),
        Avg_WinRate  = ('win_rate',     'mean'),
        Avg_Trades   = ('total_trades', 'mean'),
    ).reset_index()
    arch_stats['Avg_PnL']     = arch_stats['Avg_PnL'].apply(lambda x: f"${x:,.0f}")
    arch_stats['Avg_WinRate'] = arch_stats['Avg_WinRate'].apply(lambda x: f"{x*100:.1f}%")
    arch_stats['Avg_Trades']  = arch_stats['Avg_Trades'].apply(lambda x: f"{x:,.0f}")
    arch_stats.columns        = ['Archetype','Count','Avg Total PnL','Avg Win Rate','Avg Trades']
    st.dataframe(arch_stats, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sentiment Analysis":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("📊 Sentiment Analysis")
    st.caption("How Fear/Greed regimes affect trader performance and behavior")
    st.divider()

    tab1, tab2 = st.tabs(["Performance", "Behavior"])

    with tab1:
        st.subheader("Performance by Sentiment Regime")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(sentiment_perf['sentiment'],
                   sentiment_perf['avg_pnl'], color=colors)
            ax.set_title('Avg PnL per Trade by Sentiment')
            ax.set_ylabel('Avg PnL (USD)')
            ax.tick_params(axis='x', rotation=20)
            for i, v in enumerate(sentiment_perf['avg_pnl']):
                ax.text(i, v + 0.5, f'${v:.1f}', ha='center', fontsize=9)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(sentiment_perf['sentiment'],
                   sentiment_perf['win_rate']*100, color=colors)
            ax.set_title('Win Rate (%) by Sentiment')
            ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=20)
            for i, v in enumerate(sentiment_perf['win_rate']*100):
                ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
            st.pyplot(fig)
            plt.close()

        st.subheader("Performance Table")
        perf_display = sentiment_perf.copy()
        perf_display['avg_pnl']  = perf_display['avg_pnl'].apply(lambda x: f"${x:.1f}")
        perf_display['win_rate'] = perf_display['win_rate'].apply(lambda x: f"{x*100:.1f}%")
        perf_display['avg_size'] = perf_display['avg_size'].apply(lambda x: f"${x:,.0f}")
        perf_display.columns     = ['Sentiment','Avg PnL/Trade','Win Rate',
                                     'Total Trades','Avg Trade Size']
        st.dataframe(perf_display, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Long/Short Bias by Sentiment")
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(bias))
        ax.bar(x, bias['pct_long'],  color='#2e7d32', label='Long %')
        ax.bar(x, bias['pct_short'], bottom=bias['pct_long'],
               color='#d32f2f', label='Short %')
        ax.set_xticks(list(x))
        ax.set_xticklabels(bias['sentiment'], rotation=15)
        ax.set_ylabel('% of Directional Trades')
        ax.set_title('Long vs Short Bias by Sentiment')
        ax.legend()
        for i, (l, s) in enumerate(zip(bias['pct_long'], bias['pct_short'])):
            ax.text(i, l/2,     f'{l:.0f}%', ha='center',
                    color='white', fontsize=10, fontweight='bold')
            ax.text(i, l + s/2, f'{s:.0f}%', ha='center',
                    color='white', fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close()
        st.caption("""
        **Key insight:** Traders are systematically contrarian —
        going Long (bullish) during Fear and Short (bearish) during Greed.
        """)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Trader Explorer":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("👤 Trader Explorer")
    st.caption("Drill into any individual trader's performance profile")
    st.divider()

    selected = st.selectbox(
        "Select a Trader:",
        trader_profile['trader_id'].tolist(),
        format_func=lambda x: f"{x} — {trader_profile[trader_profile['trader_id']==x]['archetype'].values[0]}"
    )

    trader_row   = trader_profile[trader_profile['trader_id'] == selected].iloc[0]
    trader_trades = trades_closed[trades_closed['trader_id'] == selected]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total PnL",    f"${trader_row['total_pnl']:,.0f}")
    col2.metric("Win Rate",     f"{trader_row['win_rate']*100:.1f}%")
    col3.metric("Total Trades", f"{trader_row['total_trades']:,}")
    col4.metric("Archetype",    trader_row['archetype'])
    st.divider()

    st.subheader("Performance by Sentiment")
    trader_sent = trader_trades.groupby('sentiment').agg(
        avg_pnl    = ('closed_pnl', 'mean'),
        win_rate   = ('is_win',     'mean'),
        num_trades = ('closed_pnl', 'count'),
    ).reset_index()
    trader_sent['sentiment'] = pd.Categorical(
        trader_sent['sentiment'],
        categories=sentiment_order, ordered=True
    )
    trader_sent = trader_sent.sort_values('sentiment')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(trader_sent['sentiment'],
                trader_sent['avg_pnl'], color=colors[:len(trader_sent)])
    axes[0].set_title(f'{selected} — Avg PnL by Sentiment')
    axes[0].set_ylabel('Avg PnL (USD)')
    axes[0].tick_params(axis='x', rotation=20)

    axes[1].bar(trader_sent['sentiment'],
                trader_sent['win_rate']*100, color=colors[:len(trader_sent)])
    axes[1].set_title(f'{selected} — Win Rate by Sentiment')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Recent Trades")
    recent = trades_closed[trades_closed['trader_id']==selected]\
        [['date','sentiment','direction','closed_pnl','size_usd']]\
        .sort_values('date', ascending=False).head(20)
    recent['closed_pnl'] = recent['closed_pnl'].apply(lambda x: f"${x:,.2f}")
    recent['size_usd']   = recent['size_usd'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Strategy Signal":
# ══════════════════════════════════════════════════════════════════════════════
    st.title("🎯 Strategy Signal Generator")
    st.caption("Input today's Fear/Greed score → get a data-driven strategy recommendation")
    st.divider()

    fg_input = st.slider(
        "Today's Fear/Greed Index Score",
        min_value=0, max_value=100, value=50, step=1
    )

    if fg_input <= 24:
        sentiment_label = "🔴 Extreme Fear"
        color           = "red"
    elif fg_input <= 49:
        sentiment_label = "🟠 Fear"
        color           = "orange"
    elif fg_input <= 74:
        sentiment_label = "🟡 Neutral"
        color           = "yellow"
    elif fg_input <= 89:
        sentiment_label = "🟢 Greed"
        color           = "green"
    else:
        sentiment_label = "💚 Extreme Greed"
        color           = "darkgreen"

    st.subheader(f"Current Sentiment: {sentiment_label}")
    st.progress(fg_input / 100)
    st.divider()

    st.subheader("📋 Strategy Recommendation")

    if fg_input < 30:
        st.success("""
### 🟢 ACTIVATE LONG BIAS
**Signal:** Fear/Greed below 30 — Deep Fear territory

**Actions:**
- Activate Long bias on positions
- Traders with Fear win rate >80%: increase position size by up to 40%
- Developing Traders: reduce size by 50% (avg PnL only $7/trade in Fear)
- Expected win rate in Fear regime: **87.3%**
- Expected avg PnL/trade: **$112.6**

**Rationale:** Contrarian traders go Long during Fear and earn 3.1x more
than trend followers. This is the highest-activity sentiment regime.
        """)

    elif fg_input > 75:
        st.error("""
### 🔴 ACTIVATE SHORT BIAS
**Signal:** Fear/Greed above 75 — Extreme Greed territory

**Actions:**
- Activate Short bias on positions
- Reduce average position size (crowd is overleveraged)
- Expected win rate in Extreme Greed: **89.2%** for disciplined traders
- Expected avg PnL/trade: **$130.2** (highest regime)

**Rationale:** 55% of trades flip Short during Greed in our dataset.
Contrarian traders fade the rally and outperform significantly.
        """)

    elif 30 <= fg_input <= 49:
        st.warning("""
### 🟠 MILD LONG BIAS — SELECTIVE ENTRY
**Signal:** Fear/Greed 30–49 — Fear zone

**Actions:**
- Maintain Long bias but with normal position sizing
- Be selective — only enter on strong setups
- Expected win rate: **87.3%**
- Expected avg PnL/trade: **$112.6**

**Rationale:** Fear days show highest trade volume and strong returns.
Active engagement is rewarded here for skilled traders.
        """)

    elif 50 <= fg_input <= 74:
        st.info("""
### 🟡 NEUTRAL — STANDARD SIZING
**Signal:** Fear/Greed 50–74 — Neutral to Greed zone

**Actions:**
- No directional bias — trade your normal system
- Standard position sizing
- Expected win rate: **76.9–82.4%**
- Expected avg PnL/trade: **$71–85**

**Rationale:** Neutral days show the lowest avg trade size and moderate
returns. No strong sentiment edge exists in this range.
        """)

    st.divider()
    st.subheader("📊 Historical Performance at This Score Range")
    matched = sentiment_perf[
        sentiment_perf['sentiment'].astype(str).str.contains(
            sentiment_label.split(' ',1)[1] if ' ' in sentiment_label else sentiment_label
        )
    ]
    if not matched.empty:
        row = matched.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg PnL/Trade",  f"${row['avg_pnl']:.1f}")
        c2.metric("Win Rate",       f"{row['win_rate']*100:.1f}%")
        c3.metric("Typical Volume", f"{row['total_trades']:,} trades")