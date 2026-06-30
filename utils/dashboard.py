import base64
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from utils.config import BACKGROUND_IMAGE


def _img_to_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def apply_dashboard_style():
    bg_css = """
    background:
        radial-gradient(circle at 20% 20%, rgba(0, 180, 255, 0.18), transparent 25%),
        radial-gradient(circle at 80% 10%, rgba(0, 255, 170, 0.12), transparent 22%),
        linear-gradient(135deg, #07111f 0%, #081828 35%, #0a2133 70%, #0a1b2d 100%);
    """

    if BACKGROUND_IMAGE.exists():
        b64 = _img_to_base64(BACKGROUND_IMAGE)
        bg_css = f"""
        background:
            linear-gradient(rgba(6, 15, 28, 0.74), rgba(6, 15, 28, 0.74)),
            url("data:image/jpg;base64,{b64}") center center / cover no-repeat fixed;
        """

    st.markdown(
        f"""
        <style>
        .stApp {{
            {bg_css}
            color: #eaf6ff;
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 1rem;
            max-width: 1500px;
        }}

        /* 主区域文字 */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp div, .stApp span, .stApp label {{
            color: #eaf6ff;
        }}

        /* 左侧边栏单独修复 */
        section[data-testid="stSidebar"] {{
            background: rgba(245, 247, 250, 0.96) !important;
            border-right: 1px solid rgba(0, 0, 0, 0.08);
        }}

        section[data-testid="stSidebar"] * {{
            color: #182533 !important;
        }}

        section[data-testid="stSidebar"] .stButton button {{
            background: #ffffff !important;
            color: #0f1d2b !important;
            border: 1px solid #cfd8e3 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }}

        section[data-testid="stSidebar"] .stButton button:hover {{
            border-color: #4aa3ff !important;
            color: #0b5ed7 !important;
        }}

        section[data-testid="stSidebar"] .stInfo {{
            background: #eaf4ff !important;
            border: 1px solid #cfe4ff !important;
        }}

        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {{
            color: #182533 !important;
        }}

        .dashboard-title {{
            width: 100%;
            padding: 16px 24px;
            margin-bottom: 14px;
            border-radius: 16px;
            background: linear-gradient(90deg, rgba(13, 33, 54, 0.92), rgba(10, 24, 42, 0.75));
            border: 1px solid rgba(80, 180, 255, 0.30);
            box-shadow: 0 0 18px rgba(0, 180, 255, 0.12);
        }}

        .dashboard-title h1 {{
            margin: 0;
            font-size: 30px;
            font-weight: 800;
            letter-spacing: 1px;
            color: #ffffff !important;
        }}

        .dashboard-title p {{
            margin: 6px 0 0 0;
            color: #a7dfff !important;
            font-size: 14px;
        }}

        .glass-card {{
            background: rgba(8, 20, 36, 0.72);
            border: 1px solid rgba(85, 194, 255, 0.22);
            box-shadow: 0 0 16px rgba(0, 174, 255, 0.08);
            border-radius: 18px;
            padding: 14px 16px;
            backdrop-filter: blur(8px);
            margin-bottom: 14px;
        }}

        .metric-card {{
            background: linear-gradient(135deg, rgba(24, 84, 120, 0.90), rgba(8, 28, 52, 0.85));
            border: 1px solid rgba(92, 219, 255, 0.26);
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 118px;
            box-shadow: 0 0 16px rgba(0, 174, 255, 0.12);
        }}

        .metric-title {{
            font-size: 14px;
            color: #a8e7ff !important;
            margin-bottom: 12px;
        }}

        .metric-value {{
            font-size: 34px;
            font-weight: 800;
            line-height: 1.1;
            color: #ffffff !important;
        }}

        .metric-sub {{
            margin-top: 6px;
            font-size: 13px;
            color: #bfeeff !important;
        }}

        .section-title {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #eaf6ff !important;
        }}

        .small-note {{
            color: #9fdcff !important;
            font-size: 12px;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(8, 20, 36, 0.65);
            border: 1px solid rgba(85, 194, 255, 0.18);
            padding: 12px;
            border-radius: 16px;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(8, 20, 36, 0.7);
            border-radius: 12px 12px 0 0;
            border: 1px solid rgba(85, 194, 255, 0.15);
            padding: 10px 16px;
            color: #ffffff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_title():
    st.markdown(
        """
        <div class="dashboard-title">
            <h1>🌾 小麦干旱智能检测系统数据大屏</h1>
            <p>融合目标检测、ONNX 分类推理、实验指标分析与历史记录展示</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_rate_to_float(rate_value) -> float:
    if pd.isna(rate_value):
        return 0.0
    text = str(rate_value).replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def build_log_summary(log_df: pd.DataFrame) -> Dict[str, float]:
    if log_df is None or log_df.empty:
        return {
            "total_images": 0,
            "total_wheat": 0,
            "total_drought": 0,
            "avg_drought_rate": 0.0,
        }

    temp = log_df.copy()
    temp["小麦株数"] = pd.to_numeric(temp["小麦株数"], errors="coerce").fillna(0)
    temp["干旱株数"] = pd.to_numeric(temp["干旱株数"], errors="coerce").fillna(0)
    temp["干旱率数值"] = temp["干旱率"].apply(parse_rate_to_float)

    return {
        "total_images": int(len(temp)),
        "total_wheat": int(temp["小麦株数"].sum()),
        "total_drought": int(temp["干旱株数"].sum()),
        "avg_drought_rate": float(temp["干旱率数值"].mean()) if len(temp) else 0.0,
    }


def render_metric_card(title: str, value: str, sub: str):
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


def render_section_box(title: str):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


def close_section_box():
    st.markdown("</div>", unsafe_allow_html=True)