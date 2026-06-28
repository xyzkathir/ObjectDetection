import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
from datetime import datetime
import time
import pandas as pd
import threading
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO-World | Live Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e, #16213e);
    border-right: 1px solid #0f3460;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Title */
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #a445b2, #ff0066);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 10px 0 5px;
    letter-spacing: 1px;
}
.sub-title {
    text-align: center;
    color: #aab4be;
    font-size: 1rem;
    margin-bottom: 1.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(0,210,255,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d2ff;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    color: #aab4be;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #00d2ff;
    border-bottom: 1px solid rgba(0,210,255,0.3);
    padding-bottom: 6px;
    margin-bottom: 14px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,210,255,0.2);
    border-radius: 8px;
    overflow: hidden;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00d2ff, #a445b2);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: all 0.3s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,210,255,0.4);
}

/* Slider */
[data-testid="stSlider"] * { color: #00d2ff !important; }

/* Checkbox */
[data-testid="stCheckbox"] * { color: #e0e0e0 !important; }

/* Tag pills for detected objects */
.object-tag {
    display: inline-block;
    background: linear-gradient(90deg, rgba(0,210,255,0.2), rgba(164,69,178,0.2));
    border: 1px solid rgba(0,210,255,0.4);
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 0.82rem;
    color: #00d2ff;
    font-weight: 500;
}

.footer-note {
    text-align: center;
    color: #555;
    font-size: 0.75rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_FILE = os.path.join(os.path.dirname(__file__), "objects.csv")
csv_lock = threading.Lock()

# ── Session State Init ────────────────────────────────────────────────────────
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "unique_objects" not in st.session_state:
    st.session_state.unique_objects = set()
if "last_save" not in st.session_state:
    st.session_state.last_save = {}

# ── Helper Functions ──────────────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["S.No", "Date", "Time", "Object_Name"])

def get_next_sno():
    if not os.path.exists(CSV_FILE):
        return 1
    with open(CSV_FILE, "r") as f:
        return sum(1 for _ in f)  # header counts as line 1, data starts at 1

def save_detection(obj_name: str, save_interval: float):
    now = time.time()
    last = st.session_state.last_save.get(obj_name, 0)
    if (now - last) < save_interval:
        return False

    st.session_state.last_save[obj_name] = now

    with csv_lock:
        sno = get_next_sno()
        dt = datetime.now()
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                sno,
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S"),
                obj_name,
            ])
    st.session_state.total_detections += 1
    st.session_state.unique_objects.add(obj_name)
    return True

def load_csv_df() -> pd.DataFrame:
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return pd.DataFrame(columns=["S.No", "Date", "Time", "Object_Name"])
    try:
        return pd.read_csv(CSV_FILE)
    except Exception:
        return pd.DataFrame(columns=["S.No", "Date", "Time", "Object_Name"])

@st.cache_resource(show_spinner="Loading YOLO-World model…")
def load_model(model_size: str):
    size_map = {"Small (Fast)": "yolov8s-worldv2.pt",
                "Medium": "yolov8m-worldv2.pt",
                "Large (Accurate)": "yolov8l-worldv2.pt"}
    return YOLO(size_map[model_size])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 YOLO-World Settings")
    model_size = st.selectbox("Model Size", ["Small (Fast)", "Medium", "Large (Accurate)"], index=0)
    confidence = st.slider("Confidence Threshold", 0.10, 0.95, 0.30, 0.05)
    save_interval = st.slider("Save Interval (sec / object)", 1, 30, 5)
    show_labels = st.checkbox("Show Labels on Video", value=True)
    show_conf = st.checkbox("Show Confidence Scores", value=True)

    st.markdown("---")
    st.markdown("### 📁 CSV Log")
    if st.button("🗑️ Clear CSV Log"):
        init_csv()
        st.session_state.total_detections = 0
        st.session_state.unique_objects = set()
        st.session_state.last_save = {}
        st.success("Log cleared!")

    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        with open(CSV_FILE, "rb") as f:
            st.download_button("⬇️ Download CSV", f, file_name="objects.csv", mime="text/csv")

    st.markdown("---")
    st.markdown(
        "<div style='color:#555;font-size:0.75rem;'>Powered by YOLO-World + Streamlit</div>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 YOLO-World Live Object Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time detection · Auto CSV logging · WebRTC stream</div>', unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
init_csv()
model = load_model(model_size)

# ── Video Callback ────────────────────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    results = model.predict(img, conf=confidence, verbose=False)
    res = results[0]

    annotated = res.plot(labels=show_labels, conf=show_conf)

    for box in res.boxes:
        cls_id = int(box.cls[0])
        obj_name = model.names[cls_id]
        save_detection(obj_name, float(save_interval))

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ── Layout ────────────────────────────────────────────────────────────────────
col_video, col_log = st.columns([3, 2], gap="large")

with col_video:
    st.markdown('<div class="section-header">📷 Live Camera Feed</div>', unsafe_allow_html=True)

    RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    webrtc_ctx = webrtc_streamer(
        key="yolo-world-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if not webrtc_ctx.state.playing:
        st.info("👆 Click **START** above to begin live detection via your webcam.")

with col_log:
    st.markdown('<div class="section-header">📊 Detection Dashboard</div>', unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3 = st.columns(3)
    df_now = load_csv_df()
    total_rows = max(0, len(df_now) - 0)  # all data rows (header already excluded by read_csv)
    unique_count = df_now["Object_Name"].nunique() if not df_now.empty else 0

    m1.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_rows}</div>
        <div class="metric-label">Total Saved</div>
    </div>""", unsafe_allow_html=True)

    m2.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{unique_count}</div>
        <div class="metric-label">Unique Objects</div>
    </div>""", unsafe_allow_html=True)

    session_count = len(st.session_state.unique_objects)
    m3.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{session_count}</div>
        <div class="metric-label">This Session</div>
    </div>""", unsafe_allow_html=True)

    # Unique objects as tags
    if not df_now.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🏷️ Detected Classes</div>', unsafe_allow_html=True)
        tags_html = "".join(
            f'<span class="object-tag">{obj}</span>'
            for obj in sorted(df_now["Object_Name"].unique())
        )
        st.markdown(tags_html, unsafe_allow_html=True)

    # Recent detections table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Recent Detections (last 20)</div>', unsafe_allow_html=True)

    log_placeholder = st.empty()
    if not df_now.empty:
        log_placeholder.dataframe(
            df_now.tail(20).sort_index(ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        log_placeholder.info("No detections logged yet.")

# ── Auto-refresh log every 3 s when stream is active ─────────────────────────
if webrtc_ctx.state.playing:
    time.sleep(3)
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">YOLO-World © Ultralytics · Built with Streamlit · Detections auto-saved to objects.csv</div>',
    unsafe_allow_html=True,
)
