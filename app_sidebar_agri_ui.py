from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

# ============================================================
# 基础配置
# ============================================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUTPUT_DIR / "prediction_log.csv"
CAPTURED_DIR = OUTPUT_DIR / "captured_photos"
CAPTURED_DIR.mkdir(exist_ok=True)
EXPORT_DIR = OUTPUT_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)
UPLOADED_MODELS_DIR = OUTPUT_DIR / "uploaded_models"
UPLOADED_MODELS_DIR.mkdir(exist_ok=True)

MENU_ITEMS = [
    ("models", "模型选择", "folder2-open"),
    ("image_check", "图片检查", "image"),
    ("process", "过程可视化", "diagram-3"),
    ("experiment", "实验数据", "bar-chart-line"),
    ("history", "历史记录", "clock-history"),
    ("logs", "日志功能", "journal-text"),
]

MAX_BATCH_UPLOAD = 100
HISTORY_LIMIT = 50
LOG_COLUMNS = ["图像", "小麦株数", "干旱株数", "干旱率", "时间戳", "检测模型", "分类模型"]
DEFAULT_BATCH_SECONDS_PER_IMAGE = 2.0

# ============================================================
# 页面设置
# ============================================================
st.set_page_config(
    page_title="基于OpenCV的冬小麦识别与干旱监测数据大屏",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 会话状态
# ============================================================
def list_candidate_files(exts: Tuple[str, ...]) -> List[Path]:
    roots = [ROOT / "models", ROOT / "wheat_drought_runs", ROOT / "runs", ROOT / "outputs", ROOT]
    found: List[Path] = []
    for base in roots:
        if not base.exists():
            continue
        try:
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    found.append(p)
        except Exception:
            continue
    return sorted(set(found), key=lambda x: x.stat().st_mtime, reverse=True)


def list_candidate_files_in_folder(base_folder: str, exts: Tuple[str, ...]) -> List[Path]:
    if not base_folder:
        return []
    base = Path(base_folder)
    if not base.exists() or not base.is_dir():
        return []
    found: List[Path] = []
    try:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                found.append(p)
    except Exception:
        return []
    return sorted(set(found), key=lambda x: x.stat().st_mtime, reverse=True)


def latest_results_dir(roots: List[Path]) -> str:
    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            candidates.extend(p.parent for p in root.rglob("results.csv"))
        except Exception:
            continue
    if not candidates:
        return ""
    latest = max(set(candidates), key=lambda p: (p / "results.csv").stat().st_mtime)
    return str(latest)


def get_unified_eval_dirs() -> List[Path]:
    root = ROOT / "outputs" / "unified_eval"
    if not root.exists():
        return []
    dirs: List[Path] = []
    try:
        for p in root.iterdir():
            if p.is_dir() and (p / "reports" / "unified_model_comparison.csv").exists():
                dirs.append(p)
    except Exception:
        return []
    return sorted(dirs, key=lambda p: (p / "reports" / "unified_model_comparison.csv").stat().st_mtime, reverse=True)


def candidate_options(exts: Tuple[str, ...], current_path: str) -> List[str]:
    if ".pt" in exts:
        options = list(st.session_state.get("det_model_candidates", []))
    elif ".onnx" in exts or ".joblib" in exts:
        options = list(st.session_state.get("cls_model_candidates", []))
    else:
        options = []
    if current_path and current_path not in options:
        options.insert(0, current_path)
    return options


def refresh_model_candidates() -> None:
    st.session_state.det_model_candidates = [str(p) for p in list_candidate_files((".pt",))]
    st.session_state.cls_model_candidates = [str(p) for p in list_candidate_files((".onnx", ".joblib", ".pkl"))]


def save_uploaded_model(uploaded_file: Any, expected_suffix: str) -> Optional[str]:
    if uploaded_file is None:
        return None
    name = Path(uploaded_file.name).name
    if Path(name).suffix.lower() != expected_suffix:
        st.warning(f"请选择 {expected_suffix} 模型文件。")
        return None
    target = UPLOADED_MODELS_DIR / name
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def apply_model_config(det_path: str, cls_path: str, train_dir: Optional[str] = None) -> None:
    st.session_state.detector_path = det_path.strip()
    st.session_state.classifier_path = cls_path.strip()
    if train_dir is not None:
        st.session_state.train_dir = train_dir.strip()
    else:
        td = infer_train_dir_from_model(st.session_state.classifier_path)
        if td:
            st.session_state.train_dir = str(td)
    load_detector_model.clear()
    load_classifier_session.clear()
    load_svm_classifier.clear()


def infer_train_dir_from_model(path: str) -> Optional[Path]:
    p = Path(path)
    if not p.exists():
        return None
    if p.parent.name == "weights":
        return p.parent.parent
    if p.parent.name == "models" and p.parent.parent.name.startswith("unified_yolo_val"):
        return p.parent.parent
    return p.parent


def default_detector_path() -> str:
    return ""


def default_classifier_path() -> str:
    return ""


def ensure_session() -> None:
    if "active_page" not in st.session_state:
        st.session_state.active_page = "process"
    if "detector_path" not in st.session_state:
        st.session_state.detector_path = default_detector_path()
    if "classifier_path" not in st.session_state:
        st.session_state.classifier_path = default_classifier_path()
    if "train_dir" not in st.session_state:
        td = infer_train_dir_from_model(st.session_state.classifier_path)
        st.session_state.train_dir = str(td) if td else ""
    if "model_base_dir" not in st.session_state:
        st.session_state.model_base_dir = str(ROOT)
    if "folder_det_candidates" not in st.session_state:
        st.session_state.folder_det_candidates = []
    if "folder_cls_candidates" not in st.session_state:
        st.session_state.folder_cls_candidates = []
    if "det_model_candidates" not in st.session_state:
        st.session_state.det_model_candidates = [st.session_state.detector_path] if st.session_state.detector_path else []
    if "cls_model_candidates" not in st.session_state:
        st.session_state.cls_model_candidates = [st.session_state.classifier_path] if st.session_state.classifier_path else []
    if "conf_thresh" not in st.session_state:
        st.session_state.conf_thresh = 0.50
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "深色"
    if "log_df" not in st.session_state:
        if LOG_PATH.exists():
            try:
                st.session_state.log_df = pd.read_csv(LOG_PATH)
            except Exception:
                st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)
        else:
            st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)
    if "history_items" not in st.session_state:
        st.session_state.history_items = []
    if "last_process_result" not in st.session_state:
        st.session_state.last_process_result = None
    if "last_image_result" not in st.session_state:
        st.session_state.last_image_result = None
    if "confirm_clear_logs" not in st.session_state:
        st.session_state.confirm_clear_logs = False
    if "confirm_clear_sidebar" not in st.session_state:
        st.session_state.confirm_clear_sidebar = False
    if "batch_avg_seconds" not in st.session_state:
        st.session_state.batch_avg_seconds = DEFAULT_BATCH_SECONDS_PER_IMAGE
    if "detect_results_dir" not in st.session_state:
        default_detect_dir = Path("D:/pyhon/xiangmu/detect_compare_runs")
        st.session_state.detect_results_dir = str(default_detect_dir) if default_detect_dir.exists() else ""
    if "classify_results_dir" not in st.session_state:
        st.session_state.classify_results_dir = st.session_state.train_dir or latest_results_dir([ROOT / "wheat_drought_runs", ROOT / "outputs" / "train_results"])
    if "unified_eval_dir" not in st.session_state:
        unified_dirs = get_unified_eval_dirs()
        st.session_state.unified_eval_dir = str(unified_dirs[0]) if unified_dirs else ""


ensure_session()


# ============================================================
# 主题样式
# ============================================================
def inject_css(theme_mode: str) -> None:
    is_dark = theme_mode == "深色"
    bg = (
        "radial-gradient(circle at 14% 12%, rgba(34,197,94,0.16), transparent 22%),"
        "radial-gradient(circle at 90% 10%, rgba(56,189,248,0.14), transparent 24%),"
        "linear-gradient(145deg, #061018 0%, #081623 32%, #0a1d2d 68%, #0b1620 100%)"
    ) if is_dark else (
        "radial-gradient(circle at 12% 10%, rgba(34,197,94,0.12), transparent 20%),"
        "radial-gradient(circle at 88% 12%, rgba(56,189,248,0.10), transparent 22%),"
        "linear-gradient(145deg, #f1f8f4 0%, #edf7fb 44%, #f8fcff 100%)"
    )
    text = "#eaf6ff" if is_dark else "#0b2230"
    sub = "#bfeeff" if is_dark else "#365768"
    card_bg = "rgba(8, 22, 34, 0.88)" if is_dark else "rgba(255,255,255,0.94)"
    section_bg = "rgba(7, 19, 29, 0.82)" if is_dark else "rgba(255,255,255,0.92)"
    border = "rgba(90, 211, 159, 0.24)" if is_dark else "rgba(12, 102, 83, 0.18)"
    strong_border = "rgba(34,197,94,0.45)" if is_dark else "rgba(12,102,83,0.28)"
    shadow = "0 14px 34px rgba(1, 12, 18, 0.34)" if is_dark else "0 12px 28px rgba(7, 74, 128, 0.12)"
    input_bg = "rgba(8, 20, 31, 0.92)" if is_dark else "rgba(247, 252, 249, 0.98)"
    header_glow = "rgba(34,197,94,0.24)" if is_dark else "rgba(34,197,94,0.12)"

    st.markdown(
        f"""
        <style>
        :root {{
            --agri-accent: #22c55e;
            --agri-accent-soft: #7ef2a3;
            --agri-cyan: #38bdf8;
            --agri-text: {text};
            --agri-sub: {sub};
            --agri-card: {card_bg};
            --agri-section: {section_bg};
            --agri-border: {border};
            --agri-border-strong: {strong_border};
            --agri-input: {input_bg};
        }}
        .stApp {{
            background: {bg};
            color: {text};
        }}
        .block-container {{
            max-width: 1580px;
            padding-top: 0.75rem;
            padding-bottom: 1.6rem;
        }}
        h1, h2, h3, h4, h5, h6, p, div, span, label, li {{ color: {text} !important; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(5,15,23,0.98), rgba(8,20,30,0.94));
            border-right: 1px solid {border};
        }}
        [data-testid="stSidebar"] .block-container {{
            padding-top: 0.6rem;
        }}
        .hero-card, .section-card, .metric-card, .mini-card, .principle-card, .step-card, .status-strip, .empty-card {{
            border-radius: 20px;
            border: 1px solid {border};
            box-shadow: {shadow};
        }}
        .hero-card {{
            padding: 20px 24px;
            margin-bottom: 14px;
            background:
                linear-gradient(135deg, rgba(9,32,47,0.96), rgba(10,53,51,0.86)),
                radial-gradient(circle at top right, {header_glow}, transparent 40%);
            position: relative;
            overflow: hidden;
        }}
        .hero-card::after {{
            content: "";
            position: absolute;
            inset: auto -10% -55% auto;
            width: 280px;
            height: 280px;
            background: radial-gradient(circle, rgba(56,189,248,0.18), transparent 60%);
            pointer-events: none;
        }}
        .section-card {{
            padding: 16px 18px;
            margin-bottom: 14px;
            background: {section_bg};
        }}
        .metric-card {{
            padding: 16px 18px;
            min-height: 120px;
            background: linear-gradient(145deg, rgba(9,37,52,0.96), rgba(15,68,58,0.86));
        }}
        .mini-card, .principle-card, .step-card, .status-strip, .empty-card {{
            padding: 14px 16px;
            background: {card_bg};
        }}
        .title-main {{ font-size: 32px; font-weight: 800; margin-bottom: 6px; letter-spacing: 0.3px; }}
        .title-sub {{ color: {sub} !important; font-size: 14px; line-height: 1.75; max-width: 900px; }}
        .section-title {{ font-size: 20px; font-weight: 800; margin-bottom: 8px; }}
        .metric-title {{ color: {sub} !important; font-size: 14px; margin-bottom: 8px; }}
        .metric-value {{ font-size: 30px; font-weight: 800; }}
        .metric-sub {{ color: {sub} !important; font-size: 12px; margin-top: 6px; }}
        .tiny-note {{ color: {sub} !important; font-size: 13px; line-height: 1.72; }}
        .agri-green {{ color: #22c55e !important; }}
        .agri-orange {{ color: #ff9800 !important; }}
        .agri-red {{ color: #ff5252 !important; }}
        .tag-ok, .tag-info {{
            display:inline-block; padding:2px 8px; border-radius:999px;
            background: rgba(34,197,94,0.14); border:1px solid rgba(34,197,94,0.32);
            color:#cbffe0 !important; font-size:12px;
        }}
        .tag-warn {{
            display:inline-block; padding:2px 8px; border-radius:999px;
            background: rgba(255,152,0,0.15); border:1px solid rgba(255,152,0,0.32);
            color:#ffe2b1 !important; font-size:12px;
        }}
        .tag-info {{
            background: rgba(56,189,248,0.14);
            border:1px solid rgba(56,189,248,0.30);
            color:#cfefff !important;
        }}
        .status-grid {{
            display:grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }}
        .status-label {{
            font-size: 12px;
            color: {sub} !important;
            margin-bottom: 6px;
        }}
        .status-value {{
            font-size: 14px;
            font-weight: 700;
            line-height: 1.55;
            word-break: break-word;
        }}
        .nav-help {{ color:#93dfff !important; font-size:12px; line-height:1.6; }}
        .empty-title {{
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 6px;
        }}
        .empty-body {{
            color: {sub} !important;
            font-size: 13px;
            line-height: 1.7;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid {border};
            background: rgba(7,20,30,0.56);
        }}
        [data-testid="stMetric"] {{
            background: rgba(7,20,30,0.40);
            border: 1px solid {border};
            border-radius: 16px;
            padding: 10px 14px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {sub} !important;
        }}
        [data-testid="stFileUploader"] {{
            border-radius: 16px;
            border: 1px dashed {strong_border};
            background: rgba(7,20,30,0.44);
            padding: 4px;
        }}
        .stButton > button {{
            width: 100%; min-height: 46px; border-radius: 14px; font-weight: 700;
            border: 1px solid {strong_border};
            background: linear-gradient(135deg, rgba(14, 60, 44, 0.96), rgba(8, 36, 54, 0.94));
            color: #f4fbff;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: rgba(126, 242, 163, 0.65);
            box-shadow: 0 0 0 1px rgba(126, 242, 163, 0.20);
            color: #ffffff;
        }}
        .stDownloadButton > button {{
            min-height: 46px;
            border-radius: 14px;
            font-weight: 700;
            border: 1px solid {strong_border};
            background: linear-gradient(135deg, rgba(8, 36, 54, 0.96), rgba(15, 68, 58, 0.94));
            color: #f4fbff;
            width: 100%;
        }}
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
        .stTextArea textarea {{
            background: {input_bg} !important;
            color: {text} !important;
            border-radius: 14px !important;
            border: 1px solid {border} !important;
        }}
        .stRadio > div, .stSlider, .stCheckbox {{
            background: transparent;
        }}
        .st-emotion-cache-16idsys p, .st-emotion-cache-10trblm {{
            color: {text} !important;
        }}
        .sidebar-title {{ font-size: 18px; font-weight: 800; margin-top: 4px; margin-bottom: 4px; }}
        .option-menu-container {{ margin-top: 6px; }}
        @media (max-width: 900px) {{
            .block-container {{ max-width: 100%; padding-left: 0.7rem; padding-right: 0.7rem; }}
            .title-main {{ font-size: 26px; }}
            .metric-value {{ font-size: 24px; }}
            .status-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(st.session_state.theme_mode)


# ============================================================
# 通用 UI 工具函数
# ============================================================
def section(title: str, note: str = "") -> None:
    note_html = f'<div class="tiny-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            {note_html}
        """,
        unsafe_allow_html=True,
    )


def section_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def metric_card(title: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="metric-title">{title}</div>
            <div class="tiny-note">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_card(step: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="step-card">
            <div class="metric-title agri-green">{step}</div>
            <div class="tiny-note">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_strip(label: str, value: str, tag: str = "info") -> None:
    tag_class = "tag-ok" if tag == "ok" else "tag-info"
    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div style="margin-top:8px;"><span class="{tag_class}">{'已就绪' if tag == 'ok' else '展示信息'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-card">
            <div class="empty-title">{title}</div>
            <div class="empty-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def drought_color(rate_pct: float) -> str:
    if rate_pct > 30:
        return "#ff5252"
    if rate_pct > 15:
        return "#ff9800"
    return "#00c853"


def show_drought_rate(rate_str: str) -> None:
    try:
        pct = float(rate_str.replace("%", ""))
    except Exception:
        pct = 0.0
    bar_pct = max(0.0, min(1.0, pct / 100.0))
    color = drought_color(pct)
    st.progress(bar_pct)
    st.markdown(f"<div style='font-size:28px;font-weight:800;color:{color};'>{pct:.1f}%</div>", unsafe_allow_html=True)


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def safe_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits)
    exp_vals = np.exp(logits)
    denom = np.sum(exp_vals)
    if denom <= 0:
        return np.array([0.5, 0.5], dtype=np.float32)
    return exp_vals / denom


def get_model_status() -> Tuple[str, str]:
    det_ok = bool(st.session_state.detector_path and Path(st.session_state.detector_path).exists())
    cls_ok = bool(st.session_state.classifier_path and Path(st.session_state.classifier_path).exists())
    if det_ok and cls_ok:
        return "模型在线", "ok"
    if det_ok or cls_ok:
        return "部分模型待配置", "warn"
    return "模型未配置", "warn"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}小时{minutes}分钟"
    if minutes:
        return f"{minutes}分钟{sec}秒"
    return f"{sec}秒"


def estimate_batch_duration(image_count: int) -> str:
    avg_seconds = float(st.session_state.get("batch_avg_seconds", DEFAULT_BATCH_SECONDS_PER_IMAGE))
    return format_duration(image_count * max(avg_seconds, 0.5))


# ============================================================
# 模型与数据工具函数（保留原有核心能力）
# ============================================================
@st.cache_resource(show_spinner=False)
def load_detector_model(path: str):
    return YOLO(path)


@st.cache_resource(show_spinner=False)
def load_classifier_session(path: str):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(path, providers=providers)
    except Exception:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name


@st.cache_resource(show_spinner=False)
def load_svm_classifier(path: str):
    return joblib.load(path)


def preprocess_crop_for_cls(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    img_pil = Image.fromarray(gray).resize((32, 32)).convert("L")
    arr = np.array(img_pil).astype(np.float32) / 255.0
    arr = np.stack([arr] * 3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    arr = np.transpose(arr, (0, 3, 1, 2))
    return arr


def preprocess_crop_for_svm(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = Image.fromarray(gray).resize((8, 4), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)


def sigmoid(x: float) -> float:
    x = max(min(float(x), 50.0), -50.0)
    return 1.0 / (1.0 + np.exp(-x))


def predict_classifier(crop_bgr: np.ndarray, classifier: Dict[str, Any]) -> Tuple[str, float, np.ndarray, np.ndarray]:
    if classifier["type"] == "svm":
        features = preprocess_crop_for_svm(crop_bgr)
        model = classifier["model"]
        pred = int(model.predict(features)[0])
        probs: Optional[np.ndarray] = None
        if hasattr(model, "predict_proba"):
            try:
                probs = np.asarray(model.predict_proba(features)[0], dtype=np.float32)
            except Exception:
                probs = None
        if probs is None or probs.size < 2 or float(probs.sum()) <= 0:
            if hasattr(model, "decision_function"):
                score = float(np.ravel(model.decision_function(features))[0])
                p_drought = sigmoid(score)
                probs = np.array([1.0 - p_drought, p_drought], dtype=np.float32)
            else:
                probs = np.array([1.0, 0.0], dtype=np.float32) if pred == 0 else np.array([0.0, 1.0], dtype=np.float32)
        label = "drought" if pred == 1 else "control"
        label_conf = float(probs[1] if label == "drought" else probs[0])
        return label, label_conf, probs, probs

    crop_input = preprocess_crop_for_cls(crop_bgr)
    raw_output = classifier["session"].run(None, {classifier["input_name"]: crop_input})[0][0]
    probs = safe_softmax(raw_output)
    label = "drought" if probs[1] > probs[0] else "control"
    label_conf = float(probs[1] if label == "drought" else probs[0])
    return label, label_conf, probs, raw_output


def get_model_objects() -> Tuple[Any, Dict[str, Any]]:
    det_path = st.session_state.detector_path
    cls_path = st.session_state.classifier_path
    if not det_path or not Path(det_path).exists():
        raise FileNotFoundError("未找到检测模型，请先在“模型选择”页面设置检测模型路径。")
    if not cls_path or not Path(cls_path).exists():
        raise FileNotFoundError("未找到分类模型，请先在“模型选择”页面设置分类模型路径。")
    with st.spinner("正在加载检测模型与分类模型，请稍候…"):
        det_model = load_detector_model(det_path)
        if Path(cls_path).suffix.lower() in {".joblib", ".pkl"}:
            classifier = {"type": "svm", "model": load_svm_classifier(cls_path), "path": cls_path}
        else:
            cls_session, cls_input_name = load_classifier_session(cls_path)
            classifier = {"type": "onnx", "session": cls_session, "input_name": cls_input_name, "path": cls_path}
    return det_model, classifier


def append_log(row: Dict[str, Any]) -> None:
    st.session_state.log_df = pd.concat([st.session_state.log_df, pd.DataFrame([row])], ignore_index=True)
    try:
        st.session_state.log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def push_history(image_name: str, final_rgb: np.ndarray, summary: Dict[str, Any]) -> None:
    st.session_state.history_items.insert(
        0,
        {
            "图像": image_name,
            "时间戳": summary["时间戳"],
            "小麦株数": summary["小麦株数"],
            "干旱株数": summary["干旱株数"],
            "干旱率": summary["干旱率"],
            "final_rgb": final_rgb,
        },
    )
    st.session_state.history_items = st.session_state.history_items[:HISTORY_LIMIT]


def analyze_single_image(frame_bgr: np.ndarray, image_name: str, conf_thresh: float) -> Dict[str, Any]:
    det_model, classifier = get_model_objects()
    h, w = frame_bgr.shape[:2]
    results = det_model(frame_bgr, conf=conf_thresh, verbose=False)

    detection_img = frame_bgr.copy()
    final_img = frame_bgr.copy()
    detections: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []

    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
            continue
        for idx, box in enumerate(r.boxes, start=len(detections) + 1):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            det_conf = float(box.conf.item()) if hasattr(box, "conf") else 0.0
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(detection_img, f"Wheat {idx} | {det_conf:.4f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized_gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            label, label_conf, probs, raw_output = predict_classifier(crop, classifier)
            color = (0, 0, 255) if label == "drought" else (0, 255, 0)

            cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(final_img, f"{label}:{label_conf:.4f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            detections.append(
                {
                    "编号": idx,
                    "检测框": f"({x1}, {y1}) - ({x2}, {y2})",
                    "检测置信度": round(det_conf, 4),
                    "drought分数": round(float(probs[1]), 4),
                    "分类结果": label,
                    "分类置信度": round(label_conf, 4),
                }
            )
            steps.append(
                {
                    "idx": idx,
                    "bbox": (x1, y1, x2, y2),
                    "det_conf": det_conf,
                    "label": label,
                    "label_conf": label_conf,
                    "probs": probs,
                    "raw_output": raw_output,
                    "crop_rgb": to_rgb(crop),
                    "gray": gray,
                    "resized_gray": resized_gray,
                }
            )

    drought_count = sum(1 for d in detections if d["分类结果"] == "drought")
    summary = {
        "图像": image_name,
        "小麦株数": len(detections),
        "干旱株数": drought_count,
        "干旱率": f"{(drought_count / len(detections)):.1%}" if detections else "0%",
        "时间戳": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return {
        "original_rgb": to_rgb(frame_bgr),
        "detection_rgb": to_rgb(detection_img),
        "final_rgb": to_rgb(final_img),
        "detections_df": pd.DataFrame(detections),
        "steps": steps,
        "summary": summary,
    }


def load_results_csv(train_dir: str) -> Optional[pd.DataFrame]:
    p = Path(train_dir) / "results.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def collect_experiment_summaries(limit: int = 8) -> pd.DataFrame:
    roots = [ROOT / "wheat_drought_runs", ROOT / "runs", ROOT / "outputs"]
    rows: List[Dict[str, Any]] = []
    for base in roots:
        if not base.exists():
            continue
        for csv_path in base.rglob("results.csv"):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty:
                continue
            cols = list(df.columns)
            top1_col = next((c for c in cols if "accuracy_top1" in c.lower()), None)
            valloss_col = next((c for c in cols if "val/loss" in c.lower()), None)
            trainloss_col = next((c for c in cols if "train/loss" in c.lower()), None)
            row = {
                "实验目录": csv_path.parent.name,
                "路径": str(csv_path.parent),
                "最后更新时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(csv_path.stat().st_mtime)),
                "Top-1 Accuracy": f"{float(df.iloc[-1][top1_col]) * 100:.2f}%" if top1_col else "--",
                "Val Loss": f"{float(df.iloc[-1][valloss_col]):.4f}" if valloss_col else "--",
                "Train Loss": f"{float(df.iloc[-1][trainloss_col]):.4f}" if trainloss_col else "--",
                "轮次": int(len(df)),
            }
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("最后更新时间", ascending=False).head(limit)
    return result.reset_index(drop=True)


def read_table_file(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data if isinstance(data, list) else [data])
    except Exception:
        return None
    return None


def read_first_existing_table(paths: List[Path]) -> Optional[pd.DataFrame]:
    for path in paths:
        df = read_table_file(path)
        if df is not None and not df.empty:
            return df
    return None


def get_result_run_dirs(base_dir: Path) -> List[Path]:
    dirs: List[Path] = []
    if (base_dir / "results.csv").exists():
        dirs.append(base_dir)
    try:
        for p in base_dir.iterdir():
            if p.is_dir() and (p / "results.csv").exists():
                dirs.append(p)
    except Exception:
        pass
    return sorted(set(dirs), key=lambda p: (p / "results.csv").stat().st_mtime if (p / "results.csv").exists() else 0, reverse=True)


def get_classifier_result_dirs() -> List[Path]:
    roots = [ROOT / "wheat_drought_runs", ROOT / "outputs" / "train_results"]
    dirs: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "results.csv").exists():
            dirs.append(root)
        try:
            dirs.extend(p.parent for p in root.rglob("results.csv"))
        except Exception:
            continue
    return sorted(set(dirs), key=lambda p: (p / "results.csv").stat().st_mtime, reverse=True)


def is_result_dir(path_text: str) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    return path.exists() and (path / "results.csv").exists()


def find_compare_table(base_dir: Path, prefixes: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    direct_paths = [
        base_dir / "compare.csv",
        base_dir / "compare.json",
    ]
    direct_paths.extend(base_dir / f"{prefix}.csv" for prefix in prefixes)
    direct_paths.extend(base_dir / f"{prefix}.json" for prefix in prefixes)
    df = read_first_existing_table(direct_paths)
    if df is not None:
        return df
    parent = base_dir.parent if base_dir.parent != base_dir else base_dir
    for prefix in prefixes:
        for candidate in sorted(parent.glob(f"{prefix}*.csv")) + sorted(parent.glob(f"{prefix}*.json")):
            df = read_table_file(candidate)
            if df is not None and not df.empty:
                return df
    return None


def load_unified_comparison(eval_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = eval_dir / "reports" / "unified_model_comparison.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def find_unified_confusion_plot(eval_dir: Path, model_name: str) -> Path:
    return eval_dir / "plots" / f"confusion_matrix_{model_name}.png"


def get_col(df: pd.DataFrame, contains: Tuple[str, ...]) -> Optional[str]:
    lowered = {c: c.strip().lower() for c in df.columns}
    for col, low in lowered.items():
        if all(token in low for token in contains):
            return col
    return None


def format_metric_value(value: Any, as_pct: bool = True) -> str:
    try:
        val = float(value)
    except Exception:
        return "--"
    if as_pct and abs(val) <= 1.5:
        return f"{val * 100:.2f}%"
    if as_pct:
        return f"{val:.2f}%"
    return f"{val:.4f}"


def summarize_detection_run(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return "未读取到训练指标，建议检查 results.csv。"
    last = df.iloc[-1]
    precision_col = get_col(df, ("precision",))
    recall_col = get_col(df, ("recall",))
    map50_col = get_col(df, ("map50",))
    map5095_col = get_col(df, ("map50-95",)) or get_col(df, ("map50_95",))
    parts = []
    if precision_col:
        parts.append(f"Precision {format_metric_value(last[precision_col])}")
    if recall_col:
        parts.append(f"Recall {format_metric_value(last[recall_col])}")
    if map50_col:
        parts.append(f"mAP50 {format_metric_value(last[map50_col])}")
    if map5095_col:
        parts.append(f"mAP50-95 {format_metric_value(last[map5095_col])}")
    return "；".join(parts) if parts else "results.png 用于查看检测损失与 mAP 等曲线变化。"


def summarize_classify_run(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return "未读取到训练指标，建议检查 results.csv。"
    last = df.iloc[-1]
    top1_col = get_col(df, ("accuracy_top1",))
    val_loss_col = get_col(df, ("val/loss",)) or get_col(df, ("val", "loss"))
    train_loss_col = get_col(df, ("train/loss",)) or get_col(df, ("train", "loss"))
    parts = []
    if top1_col:
        parts.append(f"Top-1 {format_metric_value(last[top1_col])}")
    if val_loss_col:
        parts.append(f"Val Loss {format_metric_value(last[val_loss_col], as_pct=False)}")
    if train_loss_col:
        parts.append(f"Train Loss {format_metric_value(last[train_loss_col], as_pct=False)}")
    return "；".join(parts) if parts else "results.png 用于查看分类准确率和损失曲线变化。"


def summarize_compare_table(df: Optional[pd.DataFrame], metric_candidates: Tuple[str, ...]) -> str:
    if df is None or df.empty:
        return "暂无对比结果。"
    metric_col = next((c for c in metric_candidates if c in df.columns), None)
    if metric_col is None:
        metric_col = next((c for c in df.columns if any(k in c.lower() for k in metric_candidates)), None)
    model_col = next((c for c in ["model", "模型", "run_name"] if c in df.columns), df.columns[0])
    if metric_col is None:
        return f"共读取 {len(df)} 条模型对比记录。"
    sorted_df = df.copy()
    sorted_df[metric_col] = pd.to_numeric(sorted_df[metric_col], errors="coerce")
    best = sorted_df.sort_values(metric_col, ascending=False).iloc[0]
    return f"当前最优模型为 {best[model_col]}，{metric_col}={format_metric_value(best[metric_col])}。"


def image_analysis_text(image_name: str, run_df: Optional[pd.DataFrame], model_type: str) -> str:
    lower = image_name.lower()
    if lower == "results.png":
        return summarize_detection_run(run_df) if model_type == "detect" else summarize_classify_run(run_df)
    if "confusion_matrix_normalized" in lower:
        return "归一化混淆矩阵用于查看各类别识别比例，主对角线越高说明该类别越稳定。"
    if "confusion_matrix" in lower:
        return "混淆矩阵用于查看正确识别与误判分布，主对角线越集中表示分类或检测效果越好。"
    if "pr_curve" in lower:
        return "PR 曲线反映 Precision 与 Recall 的权衡，曲线越靠近右上方整体效果越好。"
    if "f1_curve" in lower:
        return "F1 曲线用于观察不同阈值下精确率和召回率的综合表现，峰值越高越稳定。"
    if "p_curve" in lower:
        return "Precision 曲线用于观察置信度阈值变化时误检控制能力。"
    if "r_curve" in lower:
        return "Recall 曲线用于观察置信度阈值变化时漏检控制能力。"
    return "该图用于辅助观察模型训练或评估过程。"


def render_result_image_card(title: str, image_path: Path, run_df: Optional[pd.DataFrame], model_type: str) -> None:
    section(title)
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
        st.info(image_analysis_text(image_path.name, run_df, model_type))
    else:
        st.warning(f"未读取到 {image_path.name}")
    section_end()


def get_dashboard_metrics() -> Dict[str, str]:
    log_df = st.session_state.log_df
    total_images = len(log_df) if log_df is not None else 0
    total_wheat = int(log_df["小麦株数"].sum()) if total_images else 0
    total_drought = int(log_df["干旱株数"].sum()) if total_images else 0

    acc = "--"
    valloss = "--"
    if st.session_state.train_dir:
        df = load_results_csv(st.session_state.train_dir)
        if df is not None and not df.empty:
            cols = list(df.columns)
            top1_col = next((c for c in cols if "accuracy_top1" in c.lower()), None)
            valloss_col = next((c for c in cols if "val/loss" in c.lower()), None)
            if top1_col:
                try:
                    acc = f"{float(df.iloc[-1][top1_col]) * 100:.2f}%"
                except Exception:
                    pass
            if valloss_col:
                try:
                    valloss = f"{float(df.iloc[-1][valloss_col]):.4f}"
                except Exception:
                    pass

    drought_rate = "0.00%"
    if total_wheat > 0:
        drought_rate = f"{(total_drought / total_wheat) * 100:.2f}%"

    return {
        "images": str(total_images),
        "wheat": str(total_wheat),
        "drought": str(total_drought),
        "drought_rate": drought_rate,
        "acc": acc,
        "valloss": valloss,
    }


def build_export_zip() -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if LOG_PATH.exists():
            zf.write(LOG_PATH, arcname="prediction_log.csv")
        for i, item in enumerate(st.session_state.history_items, start=1):
            rgb = item.get("final_rgb")
            if rgb is not None:
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".png", img_bgr)
                if ok:
                    zf.writestr(f"history/result_{i}_{item['图像']}.png", encoded.tobytes())
        train_dir = st.session_state.train_dir
        if train_dir:
            for name in ["results.csv", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"]:
                p = Path(train_dir) / name
                if p.exists():
                    zf.write(p, arcname=f"experiment/{name}")
    memory.seek(0)
    return memory.read()


def clear_all_history() -> None:
    st.session_state.history_items = []
    st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)
    try:
        st.session_state.log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    except Exception:
        pass


# ============================================================
# 顶部总览
# ============================================================
def render_header() -> None:
    metrics = get_dashboard_metrics()
    now_text = time.strftime("%Y-%m-%d %H:%M:%S")
    cls_name = Path(st.session_state.classifier_path).name if st.session_state.classifier_path else "未设置"
    det_name = Path(st.session_state.detector_path).name if st.session_state.detector_path else "未设置"
    model_status, status_type = get_model_status()
    status_tag = "tag-ok" if status_type == "ok" else "tag-warn"
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="title-main">🌾 基于OpenCV的冬小麦识别与干旱监测数据大屏</div>
            <div class="title-sub">融合目标检测、ONNX 分类推理、实验指标分析、过程可视化与结构化日志管理。</div>
            <div class="status-grid">
                <div class="status-strip">
                    <div class="status-label">当前时间</div>
                    <div class="status-value">{now_text}</div>
                    <div style="margin-top:8px;"><span class="tag-info">实时刷新</span></div>
                </div>
                <div class="status-strip">
                    <div class="status-label">模型状态</div>
                    <div class="status-value">{model_status}</div>
                    <div style="margin-top:8px;"><span class="{status_tag}">{'在线' if status_type == 'ok' else '待检查'}</span></div>
                </div>
                <div class="status-strip">
                    <div class="status-label">当前模型</div>
                    <div class="status-value">检测：{det_name}<br/>分类：{cls_name}</div>
                    <div style="margin-top:8px;"><span class="tag-info">当前配置</span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="medium")
    with c1:
        metric_card("累计检测图像数", metrics["images"], "日志中已记录图像")
    with c2:
        metric_card("累计识别小麦株数", metrics["wheat"], "检测框累计总数")
    with c3:
        metric_card("累计干旱株数", metrics["drought"], "分类为 drought")
    with c4:
        metric_card("实时干旱率 / 最终Val Loss", metrics["drought_rate"], f"Val Loss: {metrics['valloss']} ｜ Top-1: {metrics['acc']}")


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">全局控制区</div>', unsafe_allow_html=True)
        st.caption("调整检测阈值、主题和模型文件。")

        st.session_state.conf_thresh = st.slider(
            "检测置信度阈值",
            0.0,
            1.0,
            float(st.session_state.conf_thresh),
            0.05,
            help="阈值越高，检测越严格。",
        )

        st.session_state.theme_mode = st.radio("主题切换", ["深色", "亮色"], horizontal=True, index=0 if st.session_state.theme_mode == "深色" else 1)

        inject_css(st.session_state.theme_mode)

        st.markdown("---")
        st.markdown('<div class="sidebar-title">模型配置</div>', unsafe_allow_html=True)
        if st.button("刷新本地候选", key="sidebar_refresh_model_candidates"):
            with st.spinner("正在查找本地模型文件，请稍候…"):
                refresh_model_candidates()
            st.success(f"已找到检测模型 {len(st.session_state.det_model_candidates)} 个，分类模型 {len(st.session_state.cls_model_candidates)} 个。")
        det_options = candidate_options((".pt",), st.session_state.detector_path)
        cls_options = candidate_options((".onnx", ".joblib", ".pkl"), st.session_state.classifier_path)
        if det_options:
            chosen_det = st.selectbox("选择检测模型(.pt)", det_options, index=det_options.index(st.session_state.detector_path) if st.session_state.detector_path in det_options else 0)
            if chosen_det != st.session_state.detector_path:
                apply_model_config(chosen_det, st.session_state.classifier_path)
                st.success("检测模型已切换。")
        uploaded_det = st.file_uploader("上传检测模型(.pt)", type=["pt"], key="sidebar_upload_det")
        uploaded_det_path = save_uploaded_model(uploaded_det, ".pt")
        if uploaded_det_path and uploaded_det_path != st.session_state.detector_path:
            apply_model_config(uploaded_det_path, st.session_state.classifier_path)
            st.success("检测模型已上传并应用。")

        if cls_options:
            chosen_cls = st.selectbox("选择分类模型(.onnx/.joblib)", cls_options, index=cls_options.index(st.session_state.classifier_path) if st.session_state.classifier_path in cls_options else 0)
            if chosen_cls != st.session_state.classifier_path:
                apply_model_config(st.session_state.detector_path, chosen_cls)
                st.success("分类模型已切换。")
        uploaded_cls = st.file_uploader("上传分类模型(.onnx/.joblib)", type=["onnx", "joblib", "pkl"], key="sidebar_upload_cls")
        uploaded_cls_path = save_uploaded_model(uploaded_cls, Path(uploaded_cls.name).suffix.lower()) if uploaded_cls is not None else None
        if uploaded_cls_path and uploaded_cls_path != st.session_state.classifier_path:
            apply_model_config(st.session_state.detector_path, uploaded_cls_path)
            st.success("分类模型已上传并应用。")
        st.caption("可从候选模型中选择，也可上传模型文件。")

        st.markdown("---")
        st.markdown('<div class="sidebar-title">功能导航</div>', unsafe_allow_html=True)
        labels = [x[1] for x in MENU_ITEMS]
        icons = [x[2] for x in MENU_ITEMS]
        default_index = next((i for i, item in enumerate(MENU_ITEMS) if item[0] == st.session_state.active_page), 0)
        if option_menu is not None:
            selected_label = option_menu(
                menu_title=None,
                options=labels,
                icons=icons,
                default_index=default_index,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#00c853", "font-size": "18px"},
                    "nav-link": {
                        "font-size": "15px",
                        "font-weight": "700",
                        "text-align": "left",
                        "margin": "0 0 8px 0",
                        "padding": "12px 14px",
                        "border-radius": "12px",
                        "background-color": "rgba(9,24,41,0.76)",
                        "color": "#eaf6ff",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg, rgba(0,200,83,0.20), rgba(10, 60, 42, 0.90))",
                        "color": "#ffffff",
                        "border-left": "4px solid #00c853",
                    },
                },
            )
            selected_key = next(k for k, v, _ in MENU_ITEMS if v == selected_label)
        else:
            st.info("未安装 streamlit-option-menu，已使用内置菜单兼容显示。")
            selected_label = st.radio("功能", labels, index=default_index)
            selected_key = next(k for k, v, _ in MENU_ITEMS if v == selected_label)
        st.session_state.active_page = selected_key


# ============================================================
# 页面内容
# ============================================================
def render_image_check_page() -> None:
    section("图片检查", f"支持单图上传、拍摄和批量上传。批量上限 {MAX_BATCH_UPLOAD} 张。")
    mode = st.radio("选择检查方式", ["单图上传检测", "单图拍摄检测", "批量上传检测"], horizontal=True)

    def draw_result_panel(result: Dict[str, Any], image_name: str, frame_rgb: np.ndarray) -> None:
        row = {
            **result["summary"],
            "检测模型": Path(st.session_state.detector_path).name if st.session_state.detector_path else "",
            "分类模型": Path(st.session_state.classifier_path).name if st.session_state.classifier_path else "",
        }
        append_log(row)
        push_history(image_name, result["final_rgb"], result["summary"])
        st.session_state.last_image_result = result

        c1, c2, c3 = st.columns([1.05, 1.05, 0.9], gap="large")
        with c1:
            section("原始图像")
            st.image(frame_rgb, use_container_width=True)
            section_end()
        with c2:
            section("检测与分类结果图")
            st.image(result["final_rgb"], use_container_width=True)
            section_end()
        with c3:
            section("当前检测摘要")
            st.write(f"**图像名称：** {image_name}")
            st.write(f"**小麦株数：** {result['summary']['小麦株数']}")
            st.write(f"**干旱株数：** {result['summary']['干旱株数']}")
            st.write("**干旱率：**")
            show_drought_rate(result["summary"]["干旱率"])
            st.write(f"**检测时间：** {result['summary']['时间戳']}")
            pie_df = pd.DataFrame({"类别": ["正常", "干旱"], "数量": [max(0, result['summary']['小麦株数'] - result['summary']['干旱株数']), result['summary']['干旱株数']]})
            fig = px.pie(pie_df, names="类别", values="数量", hole=0.45, color="类别", color_discrete_map={"正常": "#00c853", "干旱": "#ff5252"})
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            section_end()

        section("检测结果明细表")
        if result["detections_df"].empty:
            st.warning("当前图像未检测到小麦目标。")
        else:
            st.dataframe(result["detections_df"], use_container_width=True, height=300)
        section_end()

    if mode == "单图上传检测":
        up = st.file_uploader("上传一张田间小麦图像", type=["png", "jpg", "jpeg"], key="single_upload")
        if up is not None:
            img = Image.open(up).convert("RGB")
            frame_rgb = np.array(img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            with st.spinner("正在执行小麦检测与干旱分类，请稍候…"):
                result = analyze_single_image(frame_bgr, up.name, st.session_state.conf_thresh)
            st.success("单图检测完成。")
            draw_result_panel(result, up.name, frame_rgb)
        else:
            render_empty_state("等待输入图像", "上传田间图像后，这里会展示原图、结果图、检测摘要、干旱率进度条以及类别占比饼图。")

    elif mode == "单图拍摄检测":
        st.info("当前模式下才加载摄像头组件，以减少页面初始化时间。")
        cam = st.camera_input("使用手机/PC 摄像头拍摄", key="single_camera")
        if cam is not None:
            img = Image.open(cam).convert("RGB")
            frame_rgb = np.array(img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            image_name = f"camera_{time.strftime('%Y%m%d_%H%M%S')}.png"
            with st.spinner("正在执行小麦检测与干旱分类，请稍候…"):
                result = analyze_single_image(frame_bgr, image_name, st.session_state.conf_thresh)
            st.success("拍摄图像检测完成。")
            draw_result_panel(result, image_name, frame_rgb)
        else:
            render_empty_state("等待拍摄图像", "拍摄后自动生成检测结果与摘要卡片。")

    else:
        batch_files = st.file_uploader("批量上传田间图像", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch_upload")
        if batch_files:
            if len(batch_files) > MAX_BATCH_UPLOAD:
                st.warning(f"当前选择了 {len(batch_files)} 张图像，系统将只处理前 {MAX_BATCH_UPLOAD} 张。")
                batch_files = batch_files[:MAX_BATCH_UPLOAD]
            st.info(f"本次将处理 {len(batch_files)} 张图像，预计耗时约 {estimate_batch_duration(len(batch_files))}。")
            summary_rows = []
            preview_cols = st.columns(2, gap="large")
            progress = st.progress(0.0)
            status_box = st.empty()
            batch_started_at = time.perf_counter()
            with st.spinner("正在批量检测，请稍候…"):
                for i, file in enumerate(batch_files, start=1):
                    elapsed = time.perf_counter() - batch_started_at
                    avg = elapsed / (i - 1) if i > 1 else float(st.session_state.get("batch_avg_seconds", DEFAULT_BATCH_SECONDS_PER_IMAGE))
                    remaining = max(len(batch_files) - i + 1, 0) * avg
                    status_box.info(f"正在处理第 {i}/{len(batch_files)} 张：{file.name} ｜ 预计剩余 {format_duration(remaining)}")
                    img = Image.open(file).convert("RGB")
                    frame_rgb = np.array(img)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    result = analyze_single_image(frame_bgr, file.name, st.session_state.conf_thresh)
                    row = {
                        **result["summary"],
                        "检测模型": Path(st.session_state.detector_path).name if st.session_state.detector_path else "",
                        "分类模型": Path(st.session_state.classifier_path).name if st.session_state.classifier_path else "",
                    }
                    append_log(row)
                    push_history(file.name, result["final_rgb"], result["summary"])
                    summary_rows.append(row)
                    with preview_cols[(i - 1) % 2]:
                        section(f"批量结果：{file.name}")
                        st.image(result["final_rgb"], use_container_width=True)
                        st.caption(f"小麦株数：{result['summary']['小麦株数']} ｜ 干旱率：{result['summary']['干旱率']}")
                        section_end()
                    progress.progress(i / len(batch_files))
            elapsed_total = time.perf_counter() - batch_started_at
            if batch_files:
                st.session_state.batch_avg_seconds = max(0.5, elapsed_total / len(batch_files))
            status_box.success(f"批量检测完成，共处理 {len(batch_files)} 张图像，用时 {format_duration(elapsed_total)}。")
            section("批量检查汇总表")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, height=320)
            section_end()
        else:
            render_empty_state("等待批量导入", f"单次最多导入 {MAX_BATCH_UPLOAD} 张图像，选择文件后会显示预计耗时。")
    section_end()


def render_experiment_page() -> None:
    section("实验数据", "按检测模型和分类模型分别查看训练结果、对比表和评估图片。")
    det_tab, cls_tab = st.tabs(["检测模型数据", "分类模型数据"])

    with det_tab:
        det_dir_text = st.text_input("检测结果文件夹", value=st.session_state.detect_results_dir, placeholder="例如：D:/pyhon/xiangmu/detect_compare_runs")
        st.session_state.detect_results_dir = det_dir_text.strip()
        det_base = Path(st.session_state.detect_results_dir) if st.session_state.detect_results_dir else None
        if not det_base or not det_base.exists():
            render_empty_state("未找到检测结果文件夹", "请填写包含 compare.csv / compare.json 或各模型 results.csv 的检测结果文件夹。")
        else:
            det_compare_df = find_compare_table(det_base, ("detect_compare", "compare"))
            section("检测模型对比表")
            if det_compare_df is not None and not det_compare_df.empty:
                st.dataframe(det_compare_df, use_container_width=True, height=260)
                st.info(summarize_compare_table(det_compare_df, ("map50", "mAP50", "map50_95", "map50-95")))
            else:
                st.warning("未读取到检测模型对比表。")
            section_end()

            det_runs = get_result_run_dirs(det_base)
            if not det_runs:
                render_empty_state("未找到检测训练目录", "请选择包含模型子目录的检测结果文件夹，子目录中应包含 results.csv。")
            else:
                run_labels = [p.name for p in det_runs]
                selected_label = st.selectbox("选择检测模型结果", run_labels)
                selected_run = det_runs[run_labels.index(selected_label)]
                det_run_df = load_results_csv(str(selected_run))

                c1, c2 = st.columns([1.1, 0.9], gap="large")
                with c1:
                    render_result_image_card("检测训练曲线", selected_run / "results.png", det_run_df, "detect")
                with c2:
                    section("检测结果摘要")
                    st.write(f"**结果目录：** {selected_run}")
                    st.write(f"**简要结论：** {summarize_detection_run(det_run_df)}")
                    if det_run_df is not None and not det_run_df.empty:
                        st.dataframe(det_run_df.tail(5), use_container_width=True, height=220)
                    section_end()

                c3, c4 = st.columns(2, gap="large")
                with c3:
                    render_result_image_card("检测混淆矩阵", selected_run / "confusion_matrix.png", det_run_df, "detect")
                with c4:
                    render_result_image_card("检测归一化混淆矩阵", selected_run / "confusion_matrix_normalized.png", det_run_df, "detect")

                curve_paths = [
                    ("PR 曲线", selected_run / "BoxPR_curve.png"),
                    ("F1 曲线", selected_run / "BoxF1_curve.png"),
                    ("Precision 曲线", selected_run / "BoxP_curve.png"),
                    ("Recall 曲线", selected_run / "BoxR_curve.png"),
                ]
                available_curves = [(title, path) for title, path in curve_paths if path.exists()]
                if available_curves:
                    section("检测评估曲线")
                    curve_cols = st.columns(2, gap="large")
                    for i, (title, path) in enumerate(available_curves):
                        with curve_cols[i % 2]:
                            st.image(str(path), caption=title, use_container_width=True)
                            st.info(image_analysis_text(path.name, det_run_df, "detect"))
                    section_end()

    with cls_tab:
        classifier_dirs = get_classifier_result_dirs()
        if not is_result_dir(st.session_state.classify_results_dir) and classifier_dirs:
            st.session_state.classify_results_dir = str(classifier_dirs[0])

        if classifier_dirs:
            candidate_labels = [str(p) for p in classifier_dirs]
            default_idx = candidate_labels.index(st.session_state.classify_results_dir) if st.session_state.classify_results_dir in candidate_labels else 0
            selected_cls_dir = st.selectbox("选择分类训练结果", candidate_labels, index=default_idx)
            st.session_state.classify_results_dir = selected_cls_dir

        cls_dir_text = st.text_input("分类结果文件夹", value=st.session_state.classify_results_dir, placeholder="例如：D:/pyhon/newA_vscode/wheat_drought_runs/exp_augmented2_s")
        st.session_state.classify_results_dir = cls_dir_text.strip()
        cls_base = Path(st.session_state.classify_results_dir) if st.session_state.classify_results_dir else None

        section("统一测试集分类模型对比表", "基于 YOLO 的 val/test 目录统一评估 SVM 与 YOLO 分类模型，口径更适合横向比较。")
        unified_dirs = get_unified_eval_dirs()
        if unified_dirs:
            unified_labels = [str(p) for p in unified_dirs]
            default_idx = unified_labels.index(st.session_state.unified_eval_dir) if st.session_state.unified_eval_dir in unified_labels else 0
            selected_unified_dir = st.selectbox("选择统一评估结果", unified_labels, index=default_idx)
            st.session_state.unified_eval_dir = selected_unified_dir
            unified_base = Path(selected_unified_dir)
            unified_df = load_unified_comparison(unified_base)
            if unified_df is not None and not unified_df.empty:
                metric_cols = [
                    "model", "family", "accuracy", "precision_macro", "recall_macro",
                    "f1_macro", "precision_drought", "recall_drought", "f1_drought",
                ]
                show_cols = [c for c in metric_cols if c in unified_df.columns]
                st.dataframe(unified_df[show_cols], use_container_width=True, height=260)
                st.info(summarize_compare_table(unified_df, ("f1_macro", "accuracy", "recall_drought")))

                available_models = [str(x) for x in unified_df["model"].tolist()] if "model" in unified_df.columns else []
                if available_models:
                    selected_model = st.selectbox("查看分类混淆矩阵", available_models)
                    cm_path = find_unified_confusion_plot(unified_base, selected_model)
                    if cm_path.exists():
                        st.image(str(cm_path), caption=f"{selected_model} confusion matrix", use_container_width=True)
            else:
                st.warning("未读取到 unified_model_comparison.csv。")
        else:
            render_empty_state("未找到统一评估结果", "请先运行 evaluate_models_on_yolo_val.py，生成 outputs/unified_eval/.../reports/unified_model_comparison.csv。")
        section_end()

        section("分类模型对比表")
        cls_compare_df = find_compare_table(cls_base, ("classifier_compare", "classify_compare")) if cls_base and cls_base.exists() else None
        if cls_compare_df is not None and not cls_compare_df.empty:
            st.dataframe(cls_compare_df, use_container_width=True, height=260)
            st.info(summarize_compare_table(cls_compare_df, ("top1_acc", "accuracy_top1", "val_loss")))
        else:
            compare_df = collect_experiment_summaries(limit=20)
            if compare_df.empty:
                st.warning("未读取到分类模型对比表。")
            else:
                st.dataframe(compare_df, use_container_width=True, height=260)
                st.info("已汇总当前项目中可读取的分类训练结果。")
        section_end()

        if not cls_base or not cls_base.exists() or not (cls_base / "results.csv").exists():
            render_empty_state("未找到分类结果文件夹", "请填写包含 results.csv、results.png 和混淆矩阵图片的分类结果文件夹。")
        else:
            cls_csv_path = cls_base / "results.csv"
            cls_run_df = load_results_csv(str(cls_base))

            c1, c2 = st.columns([1.1, 0.9], gap="large")
            with c1:
                render_result_image_card("分类训练曲线", cls_base / "results.png", cls_run_df, "classify")
            with c2:
                section("分类结果摘要")
                st.write(f"**结果目录：** {cls_base}")
                st.write(f"**简要结论：** {summarize_classify_run(cls_run_df)}")
                if cls_run_df is not None and not cls_run_df.empty:
                    st.dataframe(cls_run_df.tail(5), use_container_width=True, height=220)
                else:
                    st.warning("未读取到 results.csv")
                section_end()

            c3, c4 = st.columns(2, gap="large")
            with c3:
                render_result_image_card("分类混淆矩阵", cls_base / "confusion_matrix.png", cls_run_df, "classify")
            with c4:
                render_result_image_card("分类归一化混淆矩阵", cls_base / "confusion_matrix_normalized.png", cls_run_df, "classify")

            section("分类指标趋势")
            if cls_csv_path.exists() and cls_run_df is not None and not cls_run_df.empty:
                cols = list(cls_run_df.columns)
                loss_cols = [c for c in cols if "loss" in c.lower()]
                acc_cols = [c for c in cols if "accuracy" in c.lower()]
                if loss_cols:
                    st.line_chart(cls_run_df[loss_cols], use_container_width=True)
                if acc_cols:
                    st.line_chart(cls_run_df[acc_cols], use_container_width=True)
            else:
                st.warning("未读取到 results.csv")
            section_end()
    section_end()


def render_history_page() -> None:
    section("历史记录", f"按检测顺序保存最近 {HISTORY_LIMIT} 条识别结果，适合回看近期处理过的图像与摘要。")
    if not st.session_state.history_items:
        render_empty_state("暂无历史记录", "完成图片检查或过程可视化后，结果会自动显示在这里。")
        section_end()
        return
    for item in st.session_state.history_items:
        with st.expander(f"{item['时间戳']} ｜ {item['图像']} ｜ 干旱率：{item['干旱率']}"):
            c1, c2 = st.columns([1.1, 0.9], gap="large")
            with c1:
                st.image(item["final_rgb"], use_container_width=True)
            with c2:
                st.write(f"**图像名称：** {item['图像']}")
                st.write(f"**小麦株数：** {item['小麦株数']}")
                st.write(f"**干旱株数：** {item['干旱株数']}")
                st.write(f"**干旱率：** {item['干旱率']}")
                st.write(f"**检测时间：** {item['时间戳']}")
    section_end()


def render_logs_page() -> None:
    section("日志功能", "日志页面集中管理结构化结果，支持 CSV 导出、ZIP 打包和清空操作，方便归档实验过程。")
    log_df = st.session_state.log_df
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.download_button(
            "导出日志 CSV",
            data=log_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="prediction_log.csv",
            mime="text/csv",
            use_container_width=True,
            key="logs_export_csv",
        )
    with c2:
        zip_bytes = build_export_zip()
        st.download_button(
            "导出全部结果为ZIP",
            data=zip_bytes,
            file_name=f"wheat_drought_results_{time.strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
            key="logs_export_all_zip",
        )
    with c3:
        st.session_state.confirm_clear_logs = st.checkbox("确认清空日志与历史记录", value=st.session_state.confirm_clear_logs, key="logs_confirm_clear")
        if st.button("清除所有历史记录", use_container_width=True, disabled=not st.session_state.confirm_clear_logs):
            clear_all_history()
            st.session_state.confirm_clear_logs = False
            st.success("历史记录与日志已全部清空。")

    section("日志表")
    if log_df.empty:
        render_empty_state("日志为空", "检测完成后的结构化记录会显示在这里，可导出 CSV 或 ZIP。")
    else:
        st.dataframe(log_df, use_container_width=True, height=420)
    section_end()
    section_end()


def render_process_page() -> None:
    section("过程可视化", "展示原始图、检测框、裁剪图、灰度图、32×32 输入、分数输出和结果汇总。")
    up = st.file_uploader("上传一张小麦图像用于全过程展示", type=["png", "jpg", "jpeg"], key="process_upload")
    step_cols = st.columns(6, gap="small")
    step_texts = [
        ("步骤1", "原始图像输入"),
        ("步骤2", "YOLOv8 检测小麦"),
        ("步骤3", "裁剪单株区域"),
        ("步骤4", "构造 32×32 分类输入"),
        ("步骤5", "ONNX 输出两类分数"),
        ("步骤6", "结果图与统计汇总"),
    ]
    for col, (t, d) in zip(step_cols, step_texts):
        with col:
            step_card(t, d)

    st.markdown(
        """
        <div class="principle-card">
            <div class="section-title">原理说明</div>
            <div class="tiny-note">
            系统采用“两步法”完成小麦干旱识别。整幅田间图像先输入 YOLOv8 检测模型，模型输出每株冬小麦的边界框位置；随后系统根据检测框裁剪单株图像，并进行灰度化、缩放、归一化与维度调整，使其满足 ONNX 分类模型输入要求；分类模型输出 control 与 drought 两类分数，系统根据分数大小给出最终类别，并把结果写回原图，同时统计小麦株数、干旱株数和干旱率。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if up is None:
        render_empty_state("等待图像", "上传图像后，页面将按六步链路展开推理过程。")
        section_end()
        return

    img = Image.open(up).convert("RGB")
    frame_rgb = np.array(img)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    with st.spinner("正在生成过程可视化步骤，请稍候…"):
        result = analyze_single_image(frame_bgr, up.name, st.session_state.conf_thresh)
    st.success("过程可视化已生成。")
    st.session_state.last_process_result = result

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("步骤1：原始图像")
        st.image(result["original_rgb"], use_container_width=True)
        section_end()
    with c2:
        section("步骤2：仅显示小麦检测框")
        st.image(result["detection_rgb"], use_container_width=True)
        section_end()

    section("步骤3：检测结果明细表")
    if result["detections_df"].empty:
        st.warning("当前图像未检测到有效小麦目标。")
        section_end()
        section_end()
        return
    st.dataframe(result["detections_df"], use_container_width=True, height=260)
    section_end()

    section("步骤4-5：逐株变化过程与分类结果")
    for item in result["steps"]:
        with st.expander(f"第 {item['idx']} 株 ｜ 类别：{item['label']} ｜ 置信度：{item['label_conf']:.4f}"):
            a, b, c = st.columns(3, gap="medium")
            with a:
                st.image(item["crop_rgb"], caption="检测框裁剪图", use_container_width=True)
            with b:
                st.image(item["gray"], caption="灰度图", use_container_width=True, clamp=True)
            with c:
                st.image(item["resized_gray"], caption="32×32 分类输入", use_container_width=True, clamp=True)
            d, e = st.columns([0.55, 0.45], gap="large")
            with d:
                st.write(f"**检测框坐标：** {item['bbox']}")
                st.write(f"**检测置信度：** {item['det_conf']:.4f}")
                st.write(f"**原始输出：** {[round(float(v), 4) for v in item['raw_output']]}")
                st.write(f"**最终类别：** {item['label']}")
                st.write(f"**分类置信度：** {item['label_conf']:.4f}")
            with e:
                chart_df = pd.DataFrame({"类别": ["control", "drought"], "分数": [float(item['probs'][0]), float(item['probs'][1])]})
                st.bar_chart(chart_df.set_index("类别"), use_container_width=True)
    section_end()

    c3, c4 = st.columns([1.1, 0.9], gap="large")
    with c3:
        section("步骤6：最终成果图")
        st.image(result["final_rgb"], use_container_width=True)
        section_end()
    with c4:
        section("成果摘要")
        st.write(f"**图像名称：** {result['summary']['图像']}")
        st.write(f"**小麦株数：** {result['summary']['小麦株数']}")
        st.write(f"**干旱株数：** {result['summary']['干旱株数']}")
        st.write("**干旱率：**")
        show_drought_rate(result["summary"]["干旱率"])
        st.write(f"**检测时间：** {result['summary']['时间戳']}")
        section_end()
    section_end()


def render_models_page() -> None:
    section("模型选择", "直接选择检测模型和分类模型。")
    if st.button("刷新本地候选模型", type="primary"):
        with st.spinner("正在查找本地模型文件，请稍候…"):
            refresh_model_candidates()
        st.success(f"已找到检测模型 {len(st.session_state.det_model_candidates)} 个，分类模型 {len(st.session_state.cls_model_candidates)} 个。")

    det_candidates = candidate_options((".pt",), st.session_state.detector_path)
    cls_candidates = candidate_options((".onnx", ".joblib", ".pkl"), st.session_state.classifier_path)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.write("**检测模型(.pt)**")
        if det_candidates:
            default_idx = det_candidates.index(st.session_state.detector_path) if st.session_state.detector_path in det_candidates else 0
            chosen_det = st.selectbox("选择检测模型文件", det_candidates, index=default_idx)
        else:
            chosen_det = st.session_state.detector_path
            st.info("未发现本地 .pt 候选，可上传或手动输入路径。")
        uploaded_det = st.file_uploader("上传检测模型文件", type=["pt"], key="models_upload_det")
        uploaded_det_path = save_uploaded_model(uploaded_det, ".pt")
        if uploaded_det_path:
            chosen_det = uploaded_det_path
            st.success("检测模型已上传。")
    with c4:
        st.write("**分类模型(.onnx/.joblib)**")
        if cls_candidates:
            default_idx = cls_candidates.index(st.session_state.classifier_path) if st.session_state.classifier_path in cls_candidates else 0
            chosen_cls = st.selectbox("选择分类模型文件", cls_candidates, index=default_idx)
        else:
            chosen_cls = st.session_state.classifier_path
            st.info("未发现本地 .onnx/.joblib 候选，可上传或手动输入路径。")
        uploaded_cls = st.file_uploader("上传分类模型文件", type=["onnx", "joblib", "pkl"], key="models_upload_cls")
        uploaded_cls_path = save_uploaded_model(uploaded_cls, Path(uploaded_cls.name).suffix.lower()) if uploaded_cls is not None else None
        if uploaded_cls_path:
            chosen_cls = uploaded_cls_path
            st.success("分类模型已上传。")

    section("手动路径")
    manual_det = st.text_input("检测模型路径", value=chosen_det, placeholder="例如：D:/pyhon/newA_vscode/models/detector/best.pt")
    manual_cls = st.text_input("分类模型路径", value=chosen_cls, placeholder="例如：D:/pyhon/newA_vscode/models/classifier/best.onnx 或 outputs/unified_eval/.../models/baseline_svm_from_yolo_train.joblib")
    section_end()

    guessed_train_dir = infer_train_dir_from_model(manual_cls)
    train_dir_value = st.text_input("训练结果目录", value=str(guessed_train_dir) if guessed_train_dir else st.session_state.train_dir)

    if st.button("应用当前选择"):
        det_valid = bool(manual_det.strip() and Path(manual_det.strip()).is_file())
        cls_valid = bool(manual_cls.strip() and Path(manual_cls.strip()).is_file())
        if not det_valid:
            st.warning("检测模型路径无效，请选择或上传 .pt 文件。")
        elif not cls_valid:
            st.warning("分类模型路径无效，请选择或上传 .onnx/.joblib 文件。")
        else:
            apply_model_config(manual_det, manual_cls, train_dir_value)
            st.success("模型配置已更新。")

    c5, c6, c7 = st.columns(3, gap="large")
    with c5:
        mini_card("当前检测模型", st.session_state.detector_path or "未设置")
    with c6:
        mini_card("当前分类模型", st.session_state.classifier_path or "未设置")
    with c7:
        mini_card("当前训练目录", st.session_state.train_dir or "未设置")

    c8, c9 = st.columns(2, gap="large")
    with c8:
        section("模型候选说明")
        st.markdown(
            f"""
            <div class="tiny-note">
            1. 检测模型建议选择包含 `best.pt` 的训练权重，用于 YOLO 小麦定位。<br/>
            2. 分类模型建议选择 `best.onnx`，用于 drought / control 二分类推理。<br/>
            3. 当前候选：检测模型 {len(det_candidates)} 个，分类模型 {len(cls_candidates)} 个。<br/>
            4. 上传的模型会保存到 `outputs/uploaded_models`。
            </div>
            """,
            unsafe_allow_html=True,
        )
        section_end()
    with c9:
        section("配置建议")
        st.markdown(
            """
            <div class="tiny-note">
            1. 若实验数据页无法显示，请检查训练结果目录下是否存在 `results.csv`、`results.png` 和混淆矩阵图片。<br/>
            2. 每次应用新模型配置后，系统会自动清理缓存，下次推理将按新配置加载。<br/>
            3. 侧边栏可快速选择模型，当前页用于集中配置与核对。
            </div>
            """,
            unsafe_allow_html=True,
        )
        section_end()
    section_end()


# ============================================================
# 页面渲染
# ============================================================
def render_page() -> None:
    render_header()
    page = st.session_state.active_page
    if page == "process":
        render_process_page()
    elif page == "image_check":
        render_image_check_page()
    elif page == "experiment":
        render_experiment_page()
    elif page == "history":
        render_history_page()
    elif page == "logs":
        render_logs_page()
    else:
        render_models_page()


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    render_sidebar()
    render_page()


if __name__ == "__main__":
    main()
