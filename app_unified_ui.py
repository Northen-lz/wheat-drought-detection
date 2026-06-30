from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ============================================================
# 基础配置
# ============================================================
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUTPUT_DIR / "prediction_log.csv"
CAPTURED_DIR = OUTPUT_DIR / "captured_photos"
CAPTURED_DIR.mkdir(exist_ok=True)

PAGE_KEYS = [
    ("models", "模型选择"),
    ("process", "过程可视化"),
    ("image_check", "图片检查"),
    ("experiment", "实验数据"),
    ("history", "历史记录"),
    ("logs", "日志功能"),
]


# ============================================================
# 页面与样式
# ============================================================
st.set_page_config(
    page_title="小麦干旱智能检测系统数据大屏",
    page_icon="🌾",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(0, 190, 255, 0.12), transparent 18%),
            radial-gradient(circle at 85% 18%, rgba(0, 255, 170, 0.10), transparent 20%),
            linear-gradient(135deg, #07101a 0%, #091827 35%, #0b2030 70%, #071421 100%);
        color: #eaf6ff;
    }
    .block-container {
        max-width: 1550px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #eaf6ff !important;
    }
    .hero-card, .section-card, .metric-card, .mini-card, .principle-card {
        border-radius: 18px;
        border: 1px solid rgba(88, 194, 255, 0.18);
        background: rgba(8, 20, 36, 0.76);
        box-shadow: 0 0 18px rgba(0, 174, 255, 0.08);
    }
    .hero-card {
        padding: 20px 24px;
        margin-bottom: 14px;
        background: linear-gradient(135deg, rgba(8, 28, 52, 0.88), rgba(10, 50, 70, 0.68));
    }
    .section-card {
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .metric-card {
        padding: 16px 18px;
        min-height: 108px;
        background: linear-gradient(135deg, rgba(22, 82, 120, 0.90), rgba(8, 28, 52, 0.82));
    }
    .mini-card {
        padding: 12px 14px;
        min-height: 86px;
        background: rgba(10, 25, 42, 0.78);
    }
    .principle-card {
        padding: 14px 16px;
        min-height: 170px;
    }
    .title-main {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .title-sub {
        color: #a9def8 !important;
        font-size: 14px;
        line-height: 1.7;
    }
    .section-title {
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-title {
        color: #a9def8 !important;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 800;
    }
    .metric-sub {
        color: #bfeeff !important;
        font-size: 12px;
        margin-top: 6px;
    }
    .tiny-note {
        color: #a8dfff !important;
        font-size: 12px;
        line-height: 1.6;
    }
    .ok-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.16);
        color: #b8ffcf !important;
        border: 1px solid rgba(34, 197, 94, 0.30);
        font-size: 12px;
        margin-left: 8px;
    }
    .warn-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: rgba(251, 191, 36, 0.16);
        color: #ffe6a8 !important;
        border: 1px solid rgba(251, 191, 36, 0.30);
        font-size: 12px;
        margin-left: 8px;
    }
    .nav-tip {
        color: #8fdfff !important;
        font-size: 12px;
        margin-top: 8px;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(8, 20, 36, 0.72);
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(85, 194, 255, 0.15);
        padding: 10px 16px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(84, 194, 255, 0.20);
        background: rgba(9, 24, 41, 0.76);
        color: #eaf6ff;
        font-weight: 600;
        min-height: 44px;
    }
    .stButton > button:hover {
        border-color: rgba(84, 194, 255, 0.55);
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 工具函数
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



def infer_train_dir_from_model(path: str) -> Optional[Path]:
    p = Path(path)
    if not p.exists():
        return None
    if p.parent.name == "weights":
        return p.parent.parent
    return p.parent



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
    found = sorted(set(found), key=lambda x: x.stat().st_mtime, reverse=True)
    return found


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



def default_detector_path() -> str:
    pts = list_candidate_files((".pt",))
    for p in pts:
        sp = str(p).lower().replace("\\", "/")
        if "detect" in sp and p.name == "best.pt":
            return str(p)
    for p in pts:
        if p.name == "best.pt":
            return str(p)
    return ""



def default_classifier_path() -> str:
    onnx_files = list_candidate_files((".onnx",))
    for p in onnx_files:
        sp = str(p).lower().replace("\\", "/")
        if "augmented" in sp and p.name == "best.onnx":
            return str(p)
    for p in onnx_files:
        if p.name == "best.onnx":
            return str(p)
    return ""



def ensure_session() -> None:
    if "active_page" not in st.session_state:
        st.session_state.active_page = "models"
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
    if "conf_thresh" not in st.session_state:
        st.session_state.conf_thresh = 0.5
    if "log_df" not in st.session_state:
        if LOG_PATH.exists():
            try:
                st.session_state.log_df = pd.read_csv(LOG_PATH)
            except Exception:
                st.session_state.log_df = pd.DataFrame(columns=["图像", "小麦株数", "干旱株数", "干旱率", "时间戳", "检测模型", "分类模型"])
        else:
            st.session_state.log_df = pd.DataFrame(columns=["图像", "小麦株数", "干旱株数", "干旱率", "时间戳", "检测模型", "分类模型"])
    if "history_items" not in st.session_state:
        st.session_state.history_items = []


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



def preprocess_crop_for_cls(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    img_pil = Image.fromarray(gray).resize((32, 32)).convert("L")
    arr = np.array(img_pil).astype(np.float32) / 255.0
    arr = np.stack([arr] * 3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    arr = np.transpose(arr, (0, 3, 1, 2))
    return arr



def get_model_objects() -> Tuple[Any, Any, str]:
    det_path = st.session_state.detector_path
    cls_path = st.session_state.classifier_path
    if not det_path or not Path(det_path).exists():
        raise FileNotFoundError("未找到检测模型，请先在“模型选择”页面设置检测模型路径。")
    if not cls_path or not Path(cls_path).exists():
        raise FileNotFoundError("未找到分类模型，请先在“模型选择”页面设置分类模型路径。")
    det_model = load_detector_model(det_path)
    cls_session, cls_input_name = load_classifier_session(cls_path)
    return det_model, cls_session, cls_input_name



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
    st.session_state.history_items = st.session_state.history_items[:20]



def analyze_single_image(frame_bgr: np.ndarray, image_name: str, conf_thresh: float) -> Dict[str, Any]:
    det_model, cls_session, cls_input_name = get_model_objects()
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
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(detection_img, f"Wheat {idx} | {det_conf:.1%}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized_gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            crop_input = preprocess_crop_for_cls(crop)
            raw_output = cls_session.run(None, {cls_input_name: crop_input})[0][0]
            probs = safe_softmax(raw_output)
            label = "drought" if probs[1] > probs[0] else "control"
            label_conf = float(probs[1] if label == "drought" else probs[0])
            color = (0, 0, 255) if label == "drought" else (0, 255, 0)

            cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(final_img, f"{label}:{label_conf:.1%}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            detections.append(
                {
                    "编号": idx,
                    "检测框": f"({x1}, {y1}) - ({x2}, {y2})",
                    "检测置信度": round(det_conf, 4),
                    "control分数": round(float(probs[0]), 4),
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





def get_model_candidates():
    if "det_candidates" not in st.session_state:
        st.session_state.det_candidates = [str(p) for p in list_candidate_files((".pt",))]
    if "cls_candidates" not in st.session_state:
        st.session_state.cls_candidates = [str(p) for p in list_candidate_files((".onnx",))]
    return st.session_state.det_candidates, st.session_state.cls_candidates

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
            top1_col = next((c for c in cols if "accuracy_top1" in c), None)
            valloss_col = next((c for c in cols if c.strip().lower() == "val/loss" or "val/loss" in c), None)
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

    return {
        "images": str(total_images),
        "wheat": str(total_wheat),
        "drought": str(total_drought),
        "acc": acc,
        "valloss": valloss,
    }


# ============================================================
# 顶部区域
# ============================================================
def render_header() -> None:
    metrics = get_dashboard_metrics()
    now_text = time.strftime("%Y-%m-%d %H:%M:%S")
    cls_name = Path(st.session_state.classifier_path).name if st.session_state.classifier_path else "未设置"
    det_name = Path(st.session_state.detector_path).name if st.session_state.detector_path else "未设置"
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="title-main">🌾 小麦干旱智能检测系统数据大屏</div>
            <div class="title-sub">融合目标检测、ONNX 分类推理、实验指标分析、过程可视化与日志管理</div>
            <div class="nav-tip">当前时间：{now_text}　　模型状态：在线　　检测模型：{det_name}　　分类模型：{cls_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("累计检测图像数", metrics["images"], "日志中已记录图像")
    with c2:
        metric_card("累计识别小麦株数", metrics["wheat"], "检测框累计总数")
    with c3:
        metric_card("累计干旱株数", metrics["drought"], "分类为 drought")
    with c4:
        metric_card("实验准确率 / 最终Val Loss", metrics["acc"], f"Val Loss: {metrics['valloss']}")



def render_side_nav() -> None:
    st.markdown('<div class="section-card"><div class="section-title">功能导航</div><div class="tiny-note">按列排列，点击后仅显示当前功能页面。</div>', unsafe_allow_html=True)
    for key, label in PAGE_KEYS:
        active = st.session_state.active_page == key
        text = f"● {label}" if active else label
        if st.button(text, key=f"side_nav_{key}"):
            st.session_state.active_page = key
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 页面 1：图片检查（单图+批量）
# ============================================================
def render_image_check_page() -> None:
    section("图片检查", "图片检查统一支持单图上传、单图拍摄和批量上传。摄像头组件只在选择“拍摄检测”时加载，以减少页面初始化开销。")
    st.session_state.conf_thresh = st.slider("检测置信度阈值", 0.0, 1.0, st.session_state.conf_thresh, 0.05)

    mode = st.radio("选择检查方式", ["单图上传检测", "单图拍摄检测", "批量上传检测"], horizontal=True)

    def handle_single(source_img, image_name: str):
        frame_rgb = np.array(source_img)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        with st.spinner("正在执行小麦检测与干旱分类..."):
            result = analyze_single_image(frame_bgr, image_name, st.session_state.conf_thresh)
        row = {
            **result["summary"],
            "检测模型": Path(st.session_state.detector_path).name if st.session_state.detector_path else "",
            "分类模型": Path(st.session_state.classifier_path).name if st.session_state.classifier_path else "",
        }
        append_log(row)
        push_history(image_name, result["final_rgb"], result["summary"])

        c1, c2, c3 = st.columns([1, 1, 0.9])
        with c1:
            section("原始图像")
            st.image(frame_rgb, use_container_width=True)
            section_end()
        with c2:
            section("检测结果图")
            st.image(result["final_rgb"], use_container_width=True)
            section_end()
        with c3:
            section("当前检测摘要")
            st.write(f"**图像名称：** {image_name}")
            st.write(f"**小麦株数：** {result['summary']['小麦株数']}")
            st.write(f"**干旱株数：** {result['summary']['干旱株数']}")
            st.write(f"**干旱率：** {result['summary']['干旱率']}")
            st.write(f"**检测时间：** {result['summary']['时间戳']}")
            section_end()

        section("检测结果明细")
        if result["detections_df"].empty:
            st.warning("当前图像未检测到小麦目标。")
        else:
            st.dataframe(result["detections_df"], use_container_width=True, height=280)
        section_end()

    if mode == "单图上传检测":
        up = st.file_uploader("上传一张田间小麦图像", type=["png", "jpg", "jpeg"], key="single_upload")
        if up is not None:
            handle_single(Image.open(up).convert("RGB"), up.name)

    elif mode == "单图拍摄检测":
        st.info("点击下方摄像头组件后再拍摄，系统仅在当前模式下加载摄像头，减少页面等待时间。")
        cam = st.camera_input("使用手机/PC 摄像头拍摄", key="single_camera")
        if cam is not None:
            handle_single(Image.open(cam).convert("RGB"), f"camera_{time.strftime('%Y%m%d_%H%M%S')}.png")

    else:
        batch_files = st.file_uploader("批量上传田间图像", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch_upload")
        if batch_files:
            rows = []
            preview_cols = st.columns(2)
            for i, file in enumerate(batch_files, start=1):
                img = Image.open(file).convert("RGB")
                frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                result = analyze_single_image(frame_bgr, file.name, st.session_state.conf_thresh)
                row = {
                    **result["summary"],
                    "检测模型": Path(st.session_state.detector_path).name if st.session_state.detector_path else "",
                    "分类模型": Path(st.session_state.classifier_path).name if st.session_state.classifier_path else "",
                }
                append_log(row)
                push_history(file.name, result["final_rgb"], result["summary"])
                rows.append(row)
                with preview_cols[(i - 1) % 2]:
                    section(f"批量结果：{file.name}")
                    st.image(result["final_rgb"], use_container_width=True)
                    st.caption(f"小麦株数：{result['summary']['小麦株数']} ｜ 干旱率：{result['summary']['干旱率']}")
                    section_end()

            section("批量检查汇总表")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)
            section_end()
    section_end()


# ============================================================
# 页面 2：实验数据
# ============================================================
def render_experiment_page() -> None:
    section("实验数据", "展示训练目录中的 results.png、confusion_matrix.png、results.csv，并用曲线和表格展示模型表现。")
    train_dir = Path(st.session_state.train_dir) if st.session_state.train_dir else None
    if not train_dir or not train_dir.exists():
        st.warning("未找到训练结果目录，请先到“模型选择”页面设置分类模型或训练目录。")
        section_end()
        return

    result_png = train_dir / "results.png"
    conf_png = train_dir / "confusion_matrix.png"
    conf_norm_png = train_dir / "confusion_matrix_normalized.png"
    csv_path = train_dir / "results.csv"

    c1, c2 = st.columns(2)
    with c1:
        section("训练曲线与实验图像")
        if result_png.exists():
            st.image(str(result_png), use_container_width=True)
        else:
            st.warning("未读取到 results.png")
        section_end()
    with c2:
        section("实验结果摘要")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            cols = list(df.columns)
            top1_col = next((c for c in cols if "accuracy_top1" in c), None)
            valloss_col = next((c for c in cols if "val/loss" in c.lower()), None)
            trainloss_col = next((c for c in cols if "train/loss" in c.lower()), None)
            if not df.empty:
                if top1_col:
                    st.metric("Top-1 Accuracy", f"{float(df.iloc[-1][top1_col]) * 100:.2f}%")
                if valloss_col:
                    st.metric("Val Loss", f"{float(df.iloc[-1][valloss_col]):.4f}")
                if trainloss_col:
                    st.metric("Train Loss", f"{float(df.iloc[-1][trainloss_col]):.4f}")
        else:
            st.warning("未读取到 results.csv")
        section_end()

    c3, c4 = st.columns(2)
    with c3:
        section("混淆矩阵")
        if conf_png.exists():
            st.image(str(conf_png), use_container_width=True)
        else:
            st.warning("未读取到 confusion_matrix.png")
        section_end()
    with c4:
        section("归一化混淆矩阵")
        if conf_norm_png.exists():
            st.image(str(conf_norm_png), use_container_width=True)
        else:
            st.warning("未读取到 confusion_matrix_normalized.png")
        section_end()

    section("训练指标表与趋势")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df.tail(10), use_container_width=True, height=300)
        cols = list(df.columns)
        loss_cols = [c for c in cols if c.strip().lower() in {"train/loss", "val/loss"} or c in ["train/loss", "val/loss"]]
        top1_col = next((c for c in cols if "accuracy_top1" in c), None)
        if loss_cols:
            st.line_chart(df[loss_cols], use_container_width=True)
        if top1_col:
            st.line_chart(df[top1_col], use_container_width=True)
    else:
        st.warning("未读取到 results.csv")
    section_end()
    section_end()


# ============================================================
# 页面 4：历史记录
# ============================================================
def render_history_page() -> None:
    section("历史记录", "按检测顺序保存结果图和摘要，适合回看近期处理过的图像。")
    if not st.session_state.history_items:
        st.info("暂无历史记录，请先执行检测。")
        section_end()
        return

    for item in st.session_state.history_items:
        with st.expander(f"{item['时间戳']} ｜ {item['图像']} ｜ 干旱率：{item['干旱率']}"):
            c1, c2 = st.columns([1.1, 0.9])
            with c1:
                st.image(item["final_rgb"], use_container_width=True)
            with c2:
                st.write(f"**图像名称：** {item['图像']}")
                st.write(f"**小麦株数：** {item['小麦株数']}")
                st.write(f"**干旱株数：** {item['干旱株数']}")
                st.write(f"**干旱率：** {item['干旱率']}")
                st.write(f"**检测时间：** {item['时间戳']}")
    section_end()


# ============================================================
# 页面 5：日志功能
# ============================================================
def render_logs_page() -> None:
    section("日志功能", "日志页面单独保留结构化结果、导出、清空和统计卡片，不再混在其他页面里。")
    log_df = st.session_state.log_df
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("导出日志 CSV", key="export_logs"):
            log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
            st.success(f"日志已导出：{LOG_PATH}")
    with c2:
        if st.button("清空当前日志", key="clear_logs"):
            st.session_state.log_df = pd.DataFrame(columns=log_df.columns)
            try:
                st.session_state.log_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
            except Exception:
                pass
            st.success("日志已清空")
    with c3:
        st.write(f"**日志文件位置：** {LOG_PATH}")

    c4, c5 = st.columns(2)
    with c4:
        mini_card("字段说明", "图像、小麦株数、干旱株数、干旱率、时间戳、检测模型、分类模型。")
    with c5:
        mini_card("用途说明", "用于历史对比、成果汇总、论文截图和批量测试留档。")

    section("日志表")
    st.dataframe(st.session_state.log_df, use_container_width=True, height=420)
    section_end()
    section_end()


# ============================================================
# 页面 6：过程可视化
# ============================================================
def render_process_page() -> None:
    section("过程可视化", "把一张图从“原始输入”到“最终成果”的过程完整拆开，适合讲解系统原理和论文展示。")

    c1, c2 = st.columns([0.75, 0.25])
    with c1:
        up = st.file_uploader("上传一张小麦图像用于全过程展示", type=["png", "jpg", "jpeg"], key="process_upload")
    with c2:
        conf = st.slider("过程展示检测阈值", 0.0, 1.0, st.session_state.conf_thresh, 0.05, key="process_conf")

    st.markdown(
        """
        <div class="principle-card">
            <div class="section-title">原理说明</div>
            <div class="tiny-note">
            本系统采用“两步法”完成小麦干旱识别。整幅田间图像先输入 YOLOv8 检测模型，模型输出每株冬小麦的边界框位置；随后系统根据检测框裁剪单株图像，并进行灰度化、缩放、归一化与维度调整，使其满足 ONNX 分类模型输入要求；分类模型输出 control 与 drought 两类分数，系统根据分数大小给出最终类别，并把结果写回原图，同时统计小麦株数、干旱株数和干旱率。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step_cols = st.columns(6)
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
            mini_card(t, d)

    if up is None:
        section_end()
        return

    img = Image.open(up).convert("RGB")
    frame_rgb = np.array(img)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    result = analyze_single_image(frame_bgr, up.name, conf)

    c3, c4 = st.columns(2)
    with c3:
        section("步骤1：原始图像")
        st.image(result["original_rgb"], use_container_width=True)
        section_end()
    with c4:
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
        with st.expander(f"第 {item['idx']} 株 ｜ 类别：{item['label']} ｜ 置信度：{item['label_conf']:.1%}"):
            a, b, c = st.columns(3)
            with a:
                st.image(item["crop_rgb"], caption="检测框裁剪图", use_container_width=True)
            with b:
                st.image(item["gray"], caption="灰度图", use_container_width=True, clamp=True)
            with c:
                st.image(item["resized_gray"], caption="32×32 分类输入", use_container_width=True, clamp=True)
            d, e = st.columns([0.55, 0.45])
            with d:
                st.write(f"**检测框坐标：** {item['bbox']}")
                st.write(f"**检测置信度：** {item['det_conf']:.1%}")
                st.write(f"**原始输出：** {[round(float(v), 4) for v in item['raw_output']]}")
                st.write(f"**最终类别：** {item['label']}")
                st.write(f"**分类置信度：** {item['label_conf']:.1%}")
            with e:
                st.write("**两类分数对比**")
                chart_df = pd.DataFrame({"类别": ["control", "drought"], "分数": [float(item['probs'][0]), float(item['probs'][1])]})
                st.bar_chart(chart_df.set_index("类别"), use_container_width=True)
    section_end()

    c5, c6 = st.columns([1.1, 0.9])
    with c5:
        section("步骤6：最终成果图")
        st.image(result["final_rgb"], use_container_width=True)
        section_end()
    with c6:
        section("成果摘要")
        st.write(f"**图像名称：** {result['summary']['图像']}")
        st.write(f"**小麦株数：** {result['summary']['小麦株数']}")
        st.write(f"**干旱株数：** {result['summary']['干旱株数']}")
        st.write(f"**干旱率：** {result['summary']['干旱率']}")
        st.write(f"**检测时间：** {result['summary']['时间戳']}")
        section_end()
    section_end()


# ============================================================
# 页面 7：模型选择
# ============================================================
def render_models_page() -> None:
    section("模型选择", "改为按文件夹扫描模型。输入项目目录或模型所在目录后，系统会在该目录下递归搜索检测模型、分类模型和训练结果。")

    folder_input = st.text_input("模型/项目文件夹路径", value=st.session_state.model_base_dir, placeholder="例如：D:/pyhon/newA_vscode")
    c1, c2 = st.columns([0.22, 0.78])
    with c1:
        if st.button("扫描文件夹", type="primary", key="scan_model_folder"):
            st.session_state.model_base_dir = folder_input.strip()
            dets = [str(p) for p in list_candidate_files_in_folder(st.session_state.model_base_dir, (".pt",))]
            clss = [str(p) for p in list_candidate_files_in_folder(st.session_state.model_base_dir, (".onnx",))]
            st.session_state.folder_det_candidates = dets
            st.session_state.folder_cls_candidates = clss
            st.success(f"扫描完成：检测模型 {len(dets)} 个，分类模型 {len(clss)} 个")
    with c2:
        st.caption("提示：推荐输入项目根目录。系统不会自动加载模型，只会扫描并列出该文件夹下可用的 .pt 和 .onnx 文件。")

    det_candidates = st.session_state.folder_det_candidates
    cls_candidates = st.session_state.folder_cls_candidates

    if not det_candidates and not cls_candidates:
        st.info("请先输入文件夹路径并点击“扫描文件夹”。")
        section_end()
        return

    c3, c4 = st.columns(2)
    with c3:
        st.write("**检测模型候选**")
        if det_candidates:
            default_idx = det_candidates.index(st.session_state.detector_path) if st.session_state.detector_path in det_candidates else 0
            chosen_det = st.selectbox("选择检测模型文件", det_candidates, index=default_idx, key="folder_det_select")
        else:
            chosen_det = st.text_input("手动输入检测模型路径", value=st.session_state.detector_path, key="folder_det_manual")
    with c4:
        st.write("**分类模型候选**")
        if cls_candidates:
            default_idx = cls_candidates.index(st.session_state.classifier_path) if st.session_state.classifier_path in cls_candidates else 0
            chosen_cls = st.selectbox("选择分类模型文件", cls_candidates, index=default_idx, key="folder_cls_select")
        else:
            chosen_cls = st.text_input("手动输入分类模型路径", value=st.session_state.classifier_path, key="folder_cls_manual")

    guessed_train_dir = infer_train_dir_from_model(chosen_cls)
    train_dir_value = st.text_input("训练结果目录", value=str(guessed_train_dir) if guessed_train_dir else st.session_state.train_dir)

    c5, c6 = st.columns([0.22, 0.78])
    with c5:
        if st.button("应用当前选择", key="apply_folder_models"):
            st.session_state.detector_path = chosen_det
            st.session_state.classifier_path = chosen_cls
            st.session_state.train_dir = train_dir_value.strip()
            load_detector_model.clear()
            load_classifier_session.clear()
            st.success("模型路径已更新")
    with c6:
        st.caption("应用后，只有进入“图片检查”或“过程可视化”页面时才真正加载模型。")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        mini_card("当前检测模型", st.session_state.detector_path or "未设置")
    with cc2:
        mini_card("当前分类模型", st.session_state.classifier_path or "未设置")
    with cc3:
        mini_card("当前训练目录", st.session_state.train_dir or "未设置")

    section_end()


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    ensure_session()
    render_header()

    left, right = st.columns([0.16, 0.84], gap="medium")
    with left:
        render_side_nav()
    with right:
        page = st.session_state.active_page
        if page == "models":
            render_models_page()
        elif page == "image_check":
            render_image_check_page()
        elif page == "experiment":
            render_experiment_page()
        elif page == "history":
            render_history_page()
        elif page == "logs":
            render_logs_page()
        elif page == "process":
            render_process_page()


if __name__ == "__main__":
    main()
