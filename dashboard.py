"""
╔══════════════════════════════════════════════════════════════════╗
║   AML Risk Intelligence Dashboard                                ║
║   Graph-Based Financial Transaction Risk Detection               ║
╚══════════════════════════════════════════════════════════════════╝
Run: streamlit run streamlit_dashboard.py
"""

from pyexpat import model

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Risk Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Dark background */
  .stApp { background-color: #0a0e17; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0d1420;
    border-right: 1px solid #1e2d45;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1628 100%);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 22px 20px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4aa, #0080ff);
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #00d4aa;
    line-height: 1;
    margin-bottom: 6px;
  }
  .metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b8cae;
    font-weight: 600;
  }
  .metric-card.danger::before { background: linear-gradient(90deg, #ff4757, #ff6b35); }
  .metric-card.danger .metric-value { color: #ff4757; }
  .metric-card.warn::before { background: linear-gradient(90deg, #ffa502, #ffdd59); }
  .metric-card.warn .metric-value { color: #ffa502; }

  /* Section headers */
  .section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #1a3a5c;
  }
  .section-header .icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #0d3060, #0a1f40);
    border: 1px solid #1a4a7a;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
  }
  .section-header h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e0eaf5;
    margin: 0;
    letter-spacing: 0.03em;
  }
  .section-header span.sub {
    font-size: 0.78rem;
    color: #4a6a8a;
    font-weight: 400;
  }

  /* Model perf card */
  .model-card {
    background: #0d1f35;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 12px;
  }
  .model-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #4a9eff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
    font-weight: 600;
  }
  .perf-bar-bg {
    background: #0a1628;
    border-radius: 4px;
    height: 8px;
    margin-bottom: 6px;
    overflow: hidden;
  }
  .perf-bar { height: 8px; border-radius: 4px; }
  .perf-row {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
  }
  .perf-metric { font-size: 0.73rem; color: #6b8cae; }
  .perf-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #00d4aa;
    font-weight: 600;
  }

  /* Risk badge */
  .badge-high { color: #ff4757; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
  .badge-low  { color: #2ed573; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

  /* Table styling */
  .stDataFrame { border: 1px solid #1a3a5c !important; border-radius: 8px !important; }

  /* Plotly chart background fix */
  .js-plotly-plot { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    txn_path   = os.path.join(base, "aml_synthetic_transactions.csv")
    risk_path  = os.path.join(base, "account_risk_scores.csv")

    txn_df  = pd.read_csv(txn_path,  parse_dates=["timestamp"])
    risk_df = pd.read_csv(risk_path)
    return txn_df, risk_df

txn_df, risk_df = load_data()

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0e17",
    plot_bgcolor="#0d1f35",
    font=dict(family="IBM Plex Sans", color="#c0d4ec"),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="#122035", linecolor="#1a3a5c"),
    yaxis=dict(gridcolor="#122035", linecolor="#1a3a5c"),
)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 6px 0 24px'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.65rem;
                  color:#2a5a8a; letter-spacing:0.15em; text-transform:uppercase;
                  margin-bottom:4px'>Financial Intelligence</div>
      <div style='font-size:1.25rem; font-weight:700; color:#e0eaf5;
                  line-height:1.2'>Anti-Money Laundering Risk Detection</div>
      <div style='width:40px; height:2px; background:linear-gradient(90deg,#00d4aa,#0080ff);
                  margin-top:10px; border-radius:2px'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    sections = [
        " Dataset Overview",
        " Risk Score Table",
        "Top High-Risk Accounts",
        " Risk Score Distribution",
        " Suspicious vs Normal",
        " Network Graph",
        " Model Performance",
    ]
    nav = st.radio("", sections, label_visibility="collapsed")



 

# ── Apply sidebar filters to risk_df ──────────────────────────────────────
filtered_df = risk_df.copy()



# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 – DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if "Dataset Overview" in nav:
    st.markdown("""
    <div class="section-header">
      <div class="icon">📊</div>
      <div>
        <h2>Dataset Overview</h2>
        <span class="sub">Synthetic AML transaction dataset summary</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    n_accounts    = txn_df["sender_account"].nunique()
    n_txns        = len(txn_df)
    n_susp_txns   = txn_df["is_suspicious"].sum()
    n_susp_accts  = risk_df["is_suspicious"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value">{n_accounts:,}</div>
          <div class="metric-label">Total Accounts</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-value">{n_txns:,}</div>
          <div class="metric-label">Transactions</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card warn">
          <div class="metric-value">{n_susp_txns:,}</div>
          <div class="metric-label">Suspicious Transactions</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card danger">
          <div class="metric-value">{n_susp_accts:,}</div>
          <div class="metric-label">High-Risk Accounts</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("**Transaction Data Preview**")
        st.dataframe(
            txn_df.head(50).style.applymap(
                lambda v: "color:#ff4757; font-weight:600" if v == 1 else "",
                subset=["is_suspicious"]
            ),
            use_container_width=True, height=340
        )

    with col_r:
        st.markdown("**Transaction Type Breakdown**")
        txn_type_counts = txn_df["transaction_type"].value_counts().reset_index()
        txn_type_counts.columns = ["type", "count"]

        fig = px.bar(
            txn_type_counts, x="count", y="type", orientation="h",
            color="count", color_continuous_scale=["#0d3060", "#00d4aa"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=160,
                  showlegend=False, coloraxis_showscale=False,
                  yaxis_title=None, xaxis_title="Count")
    
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Suspicious Transaction Rate by Type**")
        susp_by_type = (
            txn_df.groupby("transaction_type")["is_suspicious"]
            .mean().mul(100).round(1).reset_index()
        )
        susp_by_type.columns = ["Type", "Susp %"]
        susp_by_type = susp_by_type.sort_values("Susp %", ascending=False)

        fig2 = px.bar(
            susp_by_type, x="Type", y="Susp %",
            color="Susp %", color_continuous_scale=["#122035", "#ff4757"]
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=170,
                   showlegend=False, coloraxis_showscale=False,
                   xaxis_title=None, yaxis_title="Susp %")
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 – RISK SCORE TABLE
# ══════════════════════════════════════════════════════════════════════════
elif "Risk Score Table" in nav:
    st.markdown("""
    <div class="section-header">
      <div class="icon">⚠️</div>
      <div>
        <h2>Risk Score Table</h2>
        <span class="sub">Per-account model predictions and ensemble scoring</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sort_col = st.selectbox("Sort by", ["ensemble_score", "lr_risk_score", "gnn_risk_score"], index=0)
    with col2:
        sort_dir = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    with col3:
        label_filter = st.multiselect("Risk Label", [0, 1], default=[0, 1])

    display_df = filtered_df[filtered_df["risk_label"].isin(label_filter)].copy()
    display_df = display_df.sort_values(sort_col, ascending=(sort_dir == "Ascending"))

    display_cols = ["account", "lr_risk_score", "gnn_risk_score", "ensemble_score", "risk_label"]
    display_df_show = display_df[display_cols].copy()

    def colour_score(val):
        if isinstance(val, float):
            r = int(255 * val)
            g = int(200 * (1 - val))
            return f"color: rgb({r},{g},80); font-family: IBM Plex Mono, monospace; font-size: 0.85rem"
        return ""

    def colour_label(val):
        if val == 1:
            return "color: #ff4757; font-weight: 700; font-family: IBM Plex Mono, monospace"
        return "color: #2ed573; font-weight: 700; font-family: IBM Plex Mono, monospace"

    styled = (
        display_df_show.style
        .applymap(colour_score, subset=["lr_risk_score", "gnn_risk_score", "ensemble_score"])
        .applymap(colour_label, subset=["risk_label"])
        .format({
            "lr_risk_score": "{:.4f}",
            "gnn_risk_score": "{:.4f}",
            "ensemble_score": "{:.4f}",
        })
    )

    st.markdown(f"**{len(display_df_show):,} accounts** matching filters")
    st.dataframe(styled, use_container_width=True, height=520)

    csv_export = display_df_show.to_csv(index=False).encode()
    st.download_button("⬇ Download filtered CSV", csv_export,
                       "filtered_risk_scores.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 – TOP HIGH-RISK ACCOUNTS
# ══════════════════════════════════════════════════════════════════════════
elif "Top High-Risk" in nav:
    st.markdown("""
    <div class="section-header">
      <div class="icon">🔴</div>
      <div>
        <h2>Top High-Risk Accounts</h2>
        <span class="sub">Ranked by ensemble risk score</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    top_n = st.slider("Number of accounts to display", 5, 50, 20)
    top_df = filtered_df.nlargest(top_n, "ensemble_score")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_df["account"], y=top_df["lr_risk_score"],
        name="Logistic Regression", marker_color="#4a9eff",
        marker_line_width=0, opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=top_df["account"], y=top_df["gnn_risk_score"],
        name="GraphSAGE", marker_color="#00d4aa",
        marker_line_width=0, opacity=0.85
    ))
    fig.add_trace(go.Scatter(
        x=top_df["account"], y=top_df["ensemble_score"],
        name="Ensemble", mode="lines+markers",
        line=dict(color="#ff4757", width=2.5),
        marker=dict(size=7, color="#ff4757", symbol="diamond")
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#ffa502",
                  annotation_text="Threshold 0.5",
                  annotation_font_color="#ffa502")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group", height=480,
        legend=dict(
            bgcolor="#0d1f35", bordercolor="#1a3a5c", borderwidth=1,
            font=dict(color="#c0d4ec")
        ),
        xaxis_title="Account", yaxis_title="Risk Score",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Risk Score Comparison — LR vs GNN**")
    scatter_fig = px.scatter(
        top_df, x="lr_risk_score", y="gnn_risk_score",
        color="ensemble_score", size="ensemble_score",
        hover_name="account",
        color_continuous_scale=["#0d3060", "#ffa502", "#ff4757"],
        labels={"lr_risk_score": "LR Score", "gnn_risk_score": "GNN Score"}
    )
    scatter_fig.update_layout(**PLOTLY_LAYOUT, height=350,
                              coloraxis_colorbar=dict(title="Ensemble", tickfont=dict(color="#c0d4ec")))
    scatter_fig.update_traces(marker_line_width=0)
    st.plotly_chart(scatter_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 – RISK SCORE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
elif "Distribution" in nav:
    st.markdown("""
    <div class="section-header">
      <div class="icon">📈</div>
      <div>
        <h2>Risk Score Distribution</h2>
        <span class="sub">Frequency analysis of ensemble risk scores</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["All Accounts — Ensemble Score", "LR · GNN · Ensemble Overlay"]
    )

    # Histogram
    normal_scores = filtered_df[filtered_df["is_suspicious"] == 0]["ensemble_score"]
    susp_scores   = filtered_df[filtered_df["is_suspicious"] == 1]["ensemble_score"]
    bins = np.linspace(0, 1, 35)

    fig.add_trace(go.Histogram(
        x=normal_scores, xbins=dict(start=0, end=1, size=1/34),
        name="Normal", marker_color="#2ed573", opacity=0.75,
        marker_line_width=0
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=susp_scores, xbins=dict(start=0, end=1, size=1/34),
        name="Suspicious", marker_color="#ff4757", opacity=0.75,
        marker_line_width=0
    ), row=1, col=1)

    # KDE overlay
    for col_name, color, label in [
        ("lr_risk_score", "#4a9eff", "LR"),
        ("gnn_risk_score", "#00d4aa", "GNN"),
        ("ensemble_score", "#ff4757", "Ensemble"),
    ]:
        vals = np.sort(filtered_df[col_name].values)
        # simple KDE via histogram density
        hist_y, hist_x = np.histogram(vals, bins=50, density=True)
        fig.add_trace(go.Scatter(
            x=(hist_x[:-1] + hist_x[1:]) / 2, y=hist_y,
            mode="lines", name=label, line=dict(color=color, width=2.5)
        ), row=1, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT, height=400, barmode="overlay",
        legend=dict(bgcolor="#0d1f35", bordercolor="#1a3a5c", borderwidth=1,
                    font=dict(color="#c0d4ec"))
    )
    fig.update_xaxes(gridcolor="#122035", linecolor="#1a3a5c")
    fig.update_yaxes(gridcolor="#122035", linecolor="#1a3a5c")
    for ann in fig.layout.annotations:
        ann.font.color = "#c0d4ec"
    st.plotly_chart(fig, use_container_width=True)

    # Box plots
    st.markdown("**Score Distribution by Risk Label**")
    box_fig = go.Figure()
    for label_val, color, name in [(0, "#2ed573", "Normal"), (1, "#ff4757", "Suspicious")]:
        sub = filtered_df[filtered_df["risk_label"] == label_val]
        for score_col, dash in [("lr_risk_score", "solid"), ("ensemble_score", "dot")]:
            box_fig.add_trace(go.Box(
                y=sub[score_col], name=f"{name} — {score_col.split('_')[0].upper()}",
                marker_color=color, line_color=color, opacity=0.8,
                boxmean=True
            ))
    box_fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          legend=dict(bgcolor="#0d1f35", bordercolor="#1a3a5c",
                                      font=dict(color="#c0d4ec")))
    st.plotly_chart(box_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 – SUSPICIOUS VS NORMAL PIE
# ══════════════════════════════════════════════════════════════════════════
elif "Suspicious vs Normal" in nav:
    st.markdown("""
    <div class="section-header">
      <div class="icon">🥧</div>
      <div>
        <h2>Suspicious vs Normal Accounts</h2>
        <span class="sub">Class distribution at account and transaction level</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # Account-level pie
    with col_l:
        acct_counts = filtered_df["risk_label"].value_counts().reset_index()
        acct_counts.columns = ["label", "count"]
        acct_counts["label"] = acct_counts["label"].map({0: "Normal", 1: "Suspicious"})

        fig1 = go.Figure(go.Pie(
            labels=acct_counts["label"],
            values=acct_counts["count"],
            hole=0.55,
            marker=dict(colors=["#2ed573", "#ff4757"],
                        line=dict(color="#0a0e17", width=3)),
            textfont=dict(family="IBM Plex Mono", size=12, color="white"),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"
        ))
        fig1.update_layout(
            **PLOTLY_LAYOUT, height=360,
            title=dict(text="Account Risk Classification",
                       font=dict(color="#c0d4ec", size=13), x=0.5),
            legend=dict(bgcolor="#0d1f35", bordercolor="#1a3a5c",
                        font=dict(color="#c0d4ec")),
            annotations=[dict(text=f"<b>{len(filtered_df)}</b><br><span style='font-size:9px'>accounts</span>",
                              x=0.5, y=0.5, font_size=18, font_color="#00d4aa",
                              showarrow=False)]
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Transaction-level pie
    with col_r:
        txn_counts = txn_df["is_suspicious"].value_counts().reset_index()
        txn_counts.columns = ["label", "count"]
        txn_counts["label"] = txn_counts["label"].map({0: "Normal", 1: "Suspicious"})

        fig2 = go.Figure(go.Pie(
            labels=txn_counts["label"],
            values=txn_counts["count"],
            hole=0.55,
            marker=dict(colors=["#4a9eff", "#ffa502"],
                        line=dict(color="#0a0e17", width=3)),
            textfont=dict(family="IBM Plex Mono", size=12, color="white"),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT, height=360,
            title=dict(text="Transaction Suspicion Flags",
                       font=dict(color="#c0d4ec", size=13), x=0.5),
            legend=dict(bgcolor="#0d1f35", bordercolor="#1a3a5c",
                        font=dict(color="#c0d4ec")),
            annotations=[dict(text=f"<b>{len(txn_df):,}</b><br><span style='font-size:9px'>transactions</span>",
                              x=0.5, y=0.5, font_size=18, font_color="#ffa502",
                              showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Sunburst: txn type × suspicious
    st.markdown("**Transaction Type × Suspicion Status**")
    sun_df = txn_df.groupby(["transaction_type", "is_suspicious"]).size().reset_index(name="count")
    sun_df["label"] = sun_df["is_suspicious"].map({0: "Normal", 1: "Suspicious"})

    sun_fig = px.sunburst(
        sun_df, path=["transaction_type", "label"], values="count",
        color="label",
        color_discrete_map={"Normal": "#4a9eff", "Suspicious": "#ff4757"}
    )
    sun_fig.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(sun_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 – INTERACTIVE NETWORK GRAPH
# ══════════════════════════════════════════════════════════════════════════
elif "Network Graph" in nav:
    st.markdown("""
    <div class="section-header">
      <div class="icon">🕸️</div>
      <div>
        <h2>Interactive Transaction Network</h2>
        <span class="sub">Account nodes · Transaction edges · Suspicious accounts highlighted</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        n_nodes = st.slider("Max accounts to display", 30, 150, 80)
    with c2:
        min_txn_amount = st.number_input("Min transaction amount ($)", 0, 50000, 0, step=500)

    # Sample top-N accounts by risk score
    top_accounts = risk_df.nlargest(n_nodes, "ensemble_score")["account"].tolist()
    sub_txn = txn_df[
        (txn_df["sender_account"].isin(top_accounts)) &
        (txn_df["receiver_account"].isin(top_accounts)) &
        (txn_df["amount"] >= min_txn_amount)
    ].copy()

    suspicious_set = set(risk_df[risk_df["risk_label"] == 1]["account"].tolist())
    score_map = dict(zip(risk_df["account"], risk_df["ensemble_score"]))

    try:
        from pyvis.network import Network

        net = Network(height="580px", width="100%", bgcolor="#0d1f35",
                      font_color="#c0d4ec", directed=True)
        net.barnes_hut(gravity=-8000, central_gravity=0.3,
                       spring_length=120, spring_strength=0.04)

        added_nodes = set()
        for acct in top_accounts:
            score = score_map.get(acct, 0)
            is_susp = acct in suspicious_set
            color  = "#ff4757" if is_susp else "#2ed573"
            size   = 18 + score * 30
            border = "#ff8c00" if is_susp else "#1a5c3a"
            net.add_node(
                acct, label=acct, color={"background": color, "border": border},
                size=size,
                title=f"Account: {acct}<br>Risk Score: {score:.3f}<br>{'🔴 SUSPICIOUS' if is_susp else '🟢 Normal'}",
                borderWidth=2 if is_susp else 1
            )
            added_nodes.add(acct)

        for _, row in sub_txn.iterrows():
            s, r = row["sender_account"], row["receiver_account"]
            if s in added_nodes and r in added_nodes:
                edge_color = "#ff4757" if row["is_suspicious"] else "#2a5a8a"
                net.add_edge(s, r, value=max(1, row["amount"] / 5000),
                             color=edge_color, title=f"${row['amount']:,.0f}")

        net.set_options("""
        {
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
            "width": 1.2
          },
          "interaction": {
            "hover": true, "navigationButtons": true,
            "tooltipDelay": 100
          },
          "physics": {"stabilization": {"iterations": 120}}
        }
        """)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            tmp_path = f.name
        net.save_graph(tmp_path)

        with open(tmp_path, "r") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=600, scrolling=False)
        os.unlink(tmp_path)

        st.markdown(
            "<div style='font-size:0.75rem; color:#4a6a8a; margin-top:8px'>"
            "🔴 Red nodes = suspicious accounts &nbsp;|&nbsp; "
            "🟢 Green nodes = normal accounts &nbsp;|&nbsp; "
            "Node size ∝ risk score &nbsp;|&nbsp; Drag to explore</div>",
            unsafe_allow_html=True
        )

    except ImportError:
        st.warning("PyVis not installed. Showing NetworkX static graph instead.")

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        G_vis = nx.DiGraph()
        for _, row in sub_txn.iterrows():
            G_vis.add_edge(row["sender_account"], row["receiver_account"],
                           weight=row["amount"])

        fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0d1f35")
        ax.set_facecolor("#0d1f35")

        pos = nx.spring_layout(G_vis, seed=42, k=0.9)
        node_colors = ["#ff4757" if n in suspicious_set else "#2ed573"
                       for n in G_vis.nodes()]
        node_sizes  = [200 + score_map.get(n, 0) * 600 for n in G_vis.nodes()]

        nx.draw_networkx_nodes(G_vis, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(G_vis, pos, ax=ax, edge_color="#2a5a8a",
                               arrows=True, arrowsize=12, alpha=0.5, width=1)
        nx.draw_networkx_labels(G_vis, pos, ax=ax, font_size=5, font_color="white")

        legend = [mpatches.Patch(color="#ff4757", label="Suspicious"),
                  mpatches.Patch(color="#2ed573", label="Normal")]
        ax.legend(handles=legend, facecolor="#0d1f35", edgecolor="#1a3a5c",
                  labelcolor="#c0d4ec")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 – MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif "Model Performance" in nav:

    st.markdown("""
    <div class="section-header">
      <div class="icon">🤖</div>
      <div>
        <h2>Model Performance</h2>
        <span class="sub">Evaluation metrics — test set results</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    models = [
        {
            "name": "Logistic Regression (Baseline)",
            "color": "#4a9eff",
            "accuracy": 0.616,
            "precision": 0.614,
            "recall": 0.672,
            "f1": 0.642,
            "roc_auc": 0.659,
        },
        {
            "name": "GraphSAGE (NumPy GNN)",
            "color": "#00d4aa",
            "accuracy": 0.448,
            "precision": 0.468,
            "recall": 0.578,
            "f1": 0.518,
            "roc_auc": 0.417,
        },
        {
            "name": "Ensemble (LR + GNN)",
            "color": "#ffa502",
            "accuracy": 0.552,
            "precision": 0.548,
            "recall": 0.719,
            "f1": 0.622,
            "roc_auc": 0.575,
        },
    ]

    col1, col2 = st.columns([2,3])

    # ── LEFT SIDE : METRIC CARDS ──
    with col1:

        for m in models:

            metrics_html = ""

            for metric_name, val in [
                ("Accuracy", m["accuracy"]),
                ("Precision", m["precision"]),
                ("Recall", m["recall"]),
                ("F1-Score", m["f1"]),
                ("ROC-AUC", m["roc_auc"]),
            ]:

                bar_w = int(val * 100)

                metrics_html += f"""
                <div class="perf-row">
                  <span class="perf-metric">{metric_name}</span>
                  <span class="perf-val">{val:.3f}</span>
                </div>

                <div class="perf-bar-bg">
                    <div class="perf-bar"
                         style="width:{bar_w}%;
                         background:linear-gradient(90deg,{m['color']}88,{m['color']});">
                    </div>
                </div>
                """

            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-name">{m['name']}</div>
                    {metrics_html}
                </div>
                """,
                unsafe_allow_html=True
            )

    # ── RIGHT SIDE : BAR CHART ──
    with col2:

        metric_names = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        metric_keys = ["accuracy","precision","recall","f1","roc_auc"]

        bar_fig = go.Figure()

        for m in models:
            bar_fig.add_trace(go.Bar(
                name=m["name"].split("(")[0].strip(),
                x=metric_names,
                y=[m[k] for k in metric_keys],
                marker_color=m["color"],
                opacity=0.85,
                marker_line_width=0
            ))

        bar_fig.add_hline(
            y=0.5,
            line_dash="dot",
            line_color="rgba(255,255,255,0.2)"
        )

        bar_fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            height=320,
            legend=dict(
                bgcolor="#0d1f35",
                bordercolor="#1a3a5c",
                font=dict(color="#c0d4ec")
            )
        )

        bar_fig.update_yaxes(range=[0,1])

        st.plotly_chart(bar_fig, use_container_width=True)