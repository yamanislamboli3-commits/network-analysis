import streamlit as st
import pandas as pd
import requests
import time
import random
import plotly.graph_objects as go
import io
import time
import threading
import tempfile
import os
from live_capture import LiveCapture, get_interfaces
from monitor import MonitorWorker
from streamlit_autorefresh import st_autorefresh

# --- AYARLAR ---
API_URL = "http://localhost:8000/predict_pcap"
API_CSV_URL = "http://localhost:8000/predict_csv"
st.set_page_config(page_title="SOC | Ağ Güvenliği Paneli", layout="wide", page_icon="🛡️")

# --- DESIGN TOKENS ---
BG = "#0A0E14"
PANEL = "#111722"
PANEL_ALT = "#0D1220"
BORDER = "#1F2937"
TEXT = "#E6EDF3"
MUTED = "#6B7785"
SAFE = "#00D9A3"
THREAT = "#FF3B5C"
ACCENT = "#FFB020"

# --- CUSTOM CSS ---
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {{
    --bg: {BG};
    --panel: {PANEL};
    --panel-alt: {PANEL_ALT};
    --border: {BORDER};
    --text: {TEXT};
    --muted: {MUTED};
    --safe: {SAFE};
    --threat: {THREAT};
    --accent: {ACCENT};
}}

.stApp {{
    background:
        radial-gradient(circle at 15% 0%, rgba(0,217,163,0.05), transparent 40%),
        radial-gradient(circle at 85% 10%, rgba(255,59,92,0.05), transparent 40%),
        var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.01em;
}}

/* --- Header signature: radar sweep --- */
.soc-header {{
    position: relative;
    padding: 1.4rem 1.6rem 1.6rem 1.6rem;
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel-alt) 100%);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 1.4rem;
    overflow: hidden;
}}
.soc-header::after {{
    content: "";
    position: absolute;
    top: 0; left: -30%;
    width: 30%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,217,163,0.12), transparent);
    animation: sweep 4.5s ease-in-out infinite;
}}
@keyframes sweep {{
    0%   {{ left: -30%; }}
    50%  {{ left: 100%; }}
    100% {{ left: 100%; }}
}}
.soc-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--safe);
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}}
.soc-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    color: var(--text);
    margin: 0;
}}
.soc-sub {{
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.3rem;
}}

/* --- Metric cards --- */
.metric-row {{ display: flex; gap: 0.9rem; margin-bottom: 1.2rem; }}
.metric-card {{
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-color, var(--muted));
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
}}
.metric-card .label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
}}
.metric-card .value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 0.15rem;
}}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
    background: transparent;
    padding: 0.6rem 1rem;
}}
.stTabs [aria-selected="true"] {{
    color: var(--safe) !important;
    border-bottom: 2px solid var(--safe) !important;
}}

/* --- Expanders (flow cards) --- */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}}
[data-testid="stExpander"] {{
    border: none !important;
}}

/* --- Buttons --- */
.stButton>button {{
    font-family: 'IBM Plex Mono', monospace;
    background: var(--safe);
    color: #06110D;
    border: none;
    border-radius: 3px;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.stButton>button:hover {{
    background: #00f0b4;
    color: #06110D;
}}

/* --- Dataframe / uploader / misc panel look --- */
[data-testid="stFileUploader"], [data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--panel);
}}

.threat-tag {{ color: var(--threat); font-weight: 700; font-family: 'IBM Plex Mono', monospace; }}
.safe-tag {{ color: var(--safe); font-weight: 700; font-family: 'IBM Plex Mono', monospace; }}

hr {{ border-color: var(--border) !important; }}
</style>
""", unsafe_allow_html=True)

# --- STATE YÖNETİMİ ---
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'total_safe' not in st.session_state:
    st.session_state.total_safe = 0
if 'total_threats' not in st.session_state:
    st.session_state.total_threats = 0


@st.cache_resource
def get_monitor_holder():
    return {"worker": None}


monitor_holder = get_monitor_holder()


def create_gauge(probability, is_attack):
    """Scope-style gauge: dark face, glowing needle-bar in the active color."""
    color = THREAT if is_attack else SAFE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'font': {'color': TEXT, 'family': 'IBM Plex Mono', 'size': 30},
                'suffix': '%'},
        title={'text': "TEHDİT SKORU", 'font': {'size': 12, 'color': MUTED, 'family': 'IBM Plex Mono'}},
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 100], 'tickcolor': MUTED, 'tickfont': {'color': MUTED, 'size': 9},
                     'linecolor': BORDER},
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': PANEL_ALT,
            'borderwidth': 1,
            'bordercolor': BORDER,
            'steps': [
                {'range': [0, 50], 'color': "rgba(0, 217, 163, 0.08)"},
                {'range': [50, 100], 'color': "rgba(255, 59, 92, 0.08)"}],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.85,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': TEXT}
    )
    return fig


def predict_pcap(uploaded_file):
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/octet-stream"
            )
        }
        response = requests.post(API_URL, files=files, timeout=300)
        if response.status_code != 200:
            st.error(response.text)
            return None
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def predict_csv(uploaded_file):
    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv"
            )
        }
        response = requests.post(API_CSV_URL, files=files, timeout=300)
        if response.status_code != 200:
            st.error(response.text)
            return None
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def metric_card_row(analyzed, safe, threats):
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card" style="--accent-color:{MUTED}">
            <div class="label">Toplam Analiz Edilen Akış</div>
            <div class="value">{analyzed:,}</div>
        </div>
        <div class="metric-card" style="--accent-color:{SAFE}">
            <div class="label">Güvenli Trafik</div>
            <div class="value" style="color:{SAFE}">{safe:,}</div>
        </div>
        <div class="metric-card" style="--accent-color:{THREAT}">
            <div class="label">Tespit Edilen Tehdit</div>
            <div class="value" style="color:{THREAT}">{threats:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- ANA PANEL ---
st.markdown("""
<div class="soc-header">
    <div class="soc-eyebrow">● LIVE MONITORING — SOC-01</div>
    <p class="soc-title">🛡️ Güvenlik Operasyon Merkezi</p>
    <p class="soc-sub">Ağ akışlarını (Network Flows) gerçek zamanlı ve geriye dönük analiz edin.</p>
</div>
""", unsafe_allow_html=True)

metric_card_row(
    st.session_state.total_analyzed,
    st.session_state.total_safe,
    st.session_state.total_threats
)

tab1, tab2 = st.tabs(["📁 PCAP/CSV Derin Analiz", "📡 Canlı Ağ Akışı (Live)"])


def display_flow_card(flow_data, result, index):
    is_success = result.get("status") == "success"
    if not is_success:
        st.error(f"Akış #{index} analiz edilemedi.")
        return

    pred = result["prediction"]
    prob = result["attack_probability"]
    is_attack = pred == "Attack"

    icon = "🔴" if is_attack else "🟢"
    status_text = "TEHDİT (ATTACK)" if is_attack else "GÜVENLİ (NORMAL)"

    with st.expander(
        f"{icon} Akış #{index} | Port: {flow_data.get('Destination Port', 'N/A')} | "
        f"Durum: {status_text} | Olasılık: %{prob*100:.1f}",
        expanded=is_attack
    ):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown("**Akış Özeti**")
            st.write(f"**Hedef Port:** {flow_data.get('Destination Port', '-')}")
            st.write(f"**Süre:** {flow_data.get('Flow Duration', '-')} ms")
            st.write(f"**Byte/s:** {flow_data.get('Flow Bytes/s', '-'):.2f}")
        with c2:
            st.markdown("**Paket İstatistikleri**")
            st.write(f"**İleri (Fwd):** {flow_data.get('Total Fwd Packets', '-')}")
            st.write(f"**Geri (Bwd):** {flow_data.get('Total Backward Packets', '-')}")
            st.write(f"**Max Uzunluk:** {flow_data.get('Max Packet Length', '-')}")
        with c3:
            st.plotly_chart(create_gauge(prob, is_attack), use_container_width=True)


def render_status_pill(running):
    if running:
        st.markdown(
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;'
            f'color:{SAFE};">● RECORDING</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;'
            f'color:{MUTED};">○ STOPPED</span>',
            unsafe_allow_html=True
        )


with tab1:
    st.subheader("📁 Offline PCAP / CSV Analysis")

    uploaded_file = st.file_uploader(
        "Upload a PCAP/PCAPNG file or a flow-features CSV",
        type=["pcap", "pcapng", "csv"]
    )

    if uploaded_file:
        st.success(f"Selected file: {uploaded_file.name}")

        if st.button("🚀 Analyze", type="primary"):
            with st.spinner("Analyzing network traffic..."):
                if uploaded_file.name.lower().endswith(".csv"):
                    result_df = predict_csv(uploaded_file)
                else:
                    result_df = predict_pcap(uploaded_file)

            if result_df is not None:
                attacks = (result_df["Prediction"] == "Attack").sum()
                normal = (result_df["Prediction"] == "Normal").sum()
                total = len(result_df)

                st.session_state.total_analyzed += total
                st.session_state.total_safe += normal
                st.session_state.total_threats += attacks

                metric_card_row(total, normal, attacks)

                st.divider()

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    height=500
                )

                csv = result_df.to_csv(index=False)

                st.download_button(
                    "⬇ Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )

with tab2:
    st.subheader("📡 Canlı Ağ Akışı (Live)")

    if "live_last_seen_update" not in st.session_state:
        st.session_state.live_last_seen_update = 0

    worker = monitor_holder["worker"]
    is_running = worker is not None and worker.running

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])

    with ctrl1:
        interfaces = get_interfaces()
        selected_iface = st.selectbox(
            "Ağ Arayüzü (Interface)",
            interfaces,
            disabled=is_running
        )
    with ctrl2:
        interval = st.number_input(
            "Aralık (sn)", min_value=2, max_value=60, value=5, step=1,
            disabled=is_running
        )
    with ctrl3:
        st.write("")
        st.write("")
        start_clicked = st.button("▶ Başlat", type="primary", disabled=is_running)
    with ctrl4:
        st.write("")
        st.write("")
        stop_clicked = st.button("■ Durdur", disabled=not is_running)

    if start_clicked and not is_running:
        new_worker = MonitorWorker(selected_iface, API_URL, interval=interval)
        new_worker.start()
        monitor_holder["worker"] = new_worker
        st.session_state.live_last_seen_update = 0
        st.rerun()

    if stop_clicked and is_running:
        worker.stop()
        st.rerun()

    st.divider()
    render_status_pill(is_running)

    if worker is not None and getattr(worker, "last_error", None):
        st.warning(f"Son hata: {worker.last_error}")

    live_placeholder = st.container()

    if is_running:
        # Poll every 2s while capture is active so new flows show up without
        # the user refreshing the page manually.
        st_autorefresh(interval=2000, key="live_autorefresh")

    with live_placeholder:
        if worker is None:
            st.info("Canlı yakalamayı başlatmak için bir arayüz seçin ve **Başlat**'a basın.")
        elif worker.latest_df is None:
            st.info("Dinleniyor... ilk akış grubunu bekliyor.")
        else:
            df = worker.latest_df

            if "Prediction" in df.columns:
                attacks = (df["Prediction"] == "Attack").sum()
                normal = (df["Prediction"] == "Normal").sum()
            else:
                attacks, normal = 0, len(df)
            total = len(df)

            # Only add to the running totals once per new batch, not on
            # every autorefresh rerun of the same batch.
            current_update_count = getattr(worker, "update_count", 0)
            if current_update_count != st.session_state.live_last_seen_update:
                st.session_state.total_analyzed += total
                st.session_state.total_safe += normal
                st.session_state.total_threats += attacks
                st.session_state.live_last_seen_update = current_update_count

            st.caption(f"Son güncelleme paketi #{current_update_count} · {total} akış")
            metric_card_row(total, normal, attacks)

            st.divider()
            st.dataframe(df, use_container_width=True, height=450)

            csv = df.to_csv(index=False)
            st.download_button(
                "⬇ Download Latest Batch",
                csv,
                "live_flows.csv",
                "text/csv",
                key="live_download"
            )