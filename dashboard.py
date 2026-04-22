"""
FL Orchestrator Agent Dashboard
Run: streamlit run dashboard.py
Auto-refreshes every 3 seconds to show live progress.
"""

import streamlit as st
import json
import os
import time

st.set_page_config(page_title="FL Orchestrator Agent", layout="wide", page_icon="🧠")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

.stApp {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: -8px;
    margin-bottom: 24px;
}

.round-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

.round-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 12px;
}

.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 12px;
}

.metric-box {
    background: #0f172a;
    border-radius: 8px;
    padding: 12px 16px;
    flex: 1;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #22d3ee;
}

.metric-value.accuracy {
    color: #34d399;
}

.metric-value.loss {
    color: #fb923c;
}

.agent-decision {
    background: #1a1a2e;
    border-left: 3px solid #8b5cf6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-top: 12px;
}

.agent-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #8b5cf6;
    font-weight: 700;
    margin-bottom: 4px;
}

.agent-reasoning {
    color: #cbd5e1;
    font-size: 0.9rem;
    line-height: 1.4;
}

.client-chips {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 6px;
}

.client-chip {
    background: #8b5cf6;
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
}

.client-chip.excluded {
    background: #334155;
    color: #64748b;
}

.status-live {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #22d3ee;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.stop-badge {
    background: #ef4444;
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

LOG_FILE = "agent_log.json"


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def render_client_chips(selected, total=10):
    chips = ""
    for i in range(total):
        if i in selected:
            chips += f'<span class="client-chip">C{i}</span>'
        else:
            chips += f'<span class="client-chip excluded">C{i}</span>'
    return chips


# Header
st.markdown('<div class="main-title">🧠 FL Orchestrator Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Autonomous federated learning with LLM-powered orchestration</div>', unsafe_allow_html=True)

# Load data
data = load_log()

if not data:
    st.markdown("""
    <div style="text-align:center; padding:60px; color:#64748b;">
        <div class="status-live" style="width:12px;height:12px;display:inline-block;"></div>
        <span style="font-family:'JetBrains Mono',monospace; font-size:1.1rem;">
            Waiting for training to start...
        </span>
        <br><br>
        <span style="font-size:0.85rem;">
            Run <code>flwr run . --stream</code> in the quickstart-pytorch directory
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    # Summary metrics at top
    latest = data[-1]
    col1, col2, col3, col4 = st.columns(4)

    latest_acc = latest.get("global_metrics", {}).get("eval_acc", 0)
    latest_loss = latest.get("global_metrics", {}).get("eval_loss", 0)
    total_rounds = len(data)
    stopped_early = any(r.get("decision", {}).get("stop_early", False) for r in data)

    with col1:
        st.metric("Rounds Completed", total_rounds)
    with col2:
        st.metric("Latest Accuracy", f"{latest_acc:.4f}")
    with col3:
        st.metric("Latest Loss", f"{latest_loss:.4f}")
    with col4:
        st.metric("Status", "⛔ Stopped Early" if stopped_early else "✅ Complete" if total_rounds > 0 else "⏳ Running")

    # Accuracy & Loss chart
    st.markdown("### 📈 Training Progress")
    col_chart1, col_chart2 = st.columns(2)

    rounds = [r["round"] for r in data]
    accs = [r.get("global_metrics", {}).get("eval_acc", 0) for r in data]
    losses = [r.get("global_metrics", {}).get("eval_loss", 0) for r in data]

    import pandas as pd

    with col_chart1:
        chart_data = pd.DataFrame({"Round": rounds, "Accuracy": accs})
        st.line_chart(chart_data.set_index("Round"), color="#34d399")

    with col_chart2:
        chart_data = pd.DataFrame({"Round": rounds, "Loss": losses})
        st.line_chart(chart_data.set_index("Round"), color="#fb923c")

    # Per-round details
    st.markdown("### 🤖 Agent Decisions Per Round")

    for entry in reversed(data):
        rnd = entry["round"]
        gm = entry.get("global_metrics", {})
        decision = entry.get("decision", {})
        selected = decision.get("selected_clients", [])
        reasoning = decision.get("reasoning", "N/A")
        stop = decision.get("stop_early", False)

        acc = gm.get("eval_acc", 0)
        loss = gm.get("eval_loss", 0)

        stop_html = ' <span class="stop-badge">EARLY STOP</span>' if stop else ""

        st.markdown(f"""
        <div class="round-card">
            <div class="round-header">Round {rnd}{stop_html}</div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value accuracy">{acc:.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Loss</div>
                    <div class="metric-value loss">{loss:.4f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Clients Selected</div>
                    <div class="metric-value">{len(selected)}/10</div>
                </div>
            </div>
            <div class="agent-decision">
                <div class="agent-label">🧠 AGENT DECISION</div>
                <div style="margin-bottom:6px;">
                    <span style="color:#94a3b8;font-size:0.85rem;">Selected clients:</span>
                    <div class="client-chips">{render_client_chips(selected)}</div>
                </div>
                <div class="agent-reasoning">{reasoning}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Auto-refresh every 3 seconds
time.sleep(3)
st.rerun()
