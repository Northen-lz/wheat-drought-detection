import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.config import (
    CAPTURED_DIR,
    PREDICTION_LOG_PATH,
    TRAIN_RESULTS_DIR,
)
from utils.inference import process_image_with_detection


# ==================== 1. 页面基础配置 ====================

st.set_page_config(
    page_title="小麦干旱智能检测系统",
    page_icon="🌾",
    layout="wide",
)

LOG_COLUMNS = ["图像", "小麦株数", "干旱株数", "干旱率", "时间戳"]


# ==================== 2. 日志工具函数 ====================

def init_log():
    if "log_df" not in st.session_state:
        if PREDICTION_LOG_PATH.exists():
            try:
                st.session_state.log_df = pd.read_csv(PREDICTION_LOG_PATH)
            except Exception:
                st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)
        else:
            st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)


def save_log_to_disk():
    init_log()
    st.session_state.log_df.to_csv(PREDICTION_LOG_PATH, index=False, encoding="utf-8-sig")


def update_log(data, path=None):
    init_log()

    row = {
        "图像": os.path.basename(path) if path else data["图像"],
        "小麦株数": data["小麦株数"],
        "干旱株数": data["干旱株数"],
        "干旱率": data["干旱率"],
        "时间戳": data["时间戳"],
    }

    st.session_state.log_df = pd.concat(
        [st.session_state.log_df, pd.DataFrame([row])],
        ignore_index=True,
    )

    save_log_to_disk()


# ==================== 3. 训练结果展示 ====================

def show_training_results():
    st.header("训练实验数据查看")

    with st.expander("训练结果分析", expanded=True):
        st.subheader("数据分析介绍")
        st.markdown(
            """
            **训练实验总结**：
            - **训练轮数**：50 epochs
            - **分类任务**：判断 drought / control
            - **输入方式**：先检测小麦，再对裁剪小图做分类
            - **建议**：继续增加训练轮数、优化数据增强、尝试更强的分类模型
            """
        )

        results_png_path = TRAIN_RESULTS_DIR / "results.png"
        confusion_path = TRAIN_RESULTS_DIR / "confusion_matrix.png"
        csv_path = TRAIN_RESULTS_DIR / "results.csv"

        col1, col2 = st.columns(2)

        with col1:
            if results_png_path.exists():
                st.image(str(results_png_path), caption="训练曲线 results.png")
            else:
                st.warning(f"未找到：{results_png_path}")

        with col2:
            if confusion_path.exists():
                st.image(str(confusion_path), caption="混淆矩阵 confusion_matrix.png")
            else:
                st.warning(f"未找到：{confusion_path}")

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                st.subheader("详细训练指标表")
                st.dataframe(df.tail(10), use_container_width=True)

                loss_cols = [c for c in ["train/loss", "val/loss"] if c in df.columns]
                if loss_cols:
                    st.subheader("Loss 曲线")
                    st.line_chart(df[loss_cols], use_container_width=True)

                if "metrics/accuracy_top1" in df.columns:
                    st.subheader("Top-1 Accuracy 曲线")
                    st.line_chart(df["metrics/accuracy_top1"], use_container_width=True)
            except Exception as e:
                st.error(f"读取 results.csv 失败：{e}")
        else:
            st.warning(f"未找到：{csv_path}")


# ==================== 4. 主界面 ====================

def main():
    init_log()

    st.title("🌾 小麦干旱智能检测系统")
    st.markdown("**流程：先检测小麦 → 再判断每株是否干旱**")

    tab = st.sidebar.radio("功能导航", ["检测模式", "训练数据查看"])

    if tab == "检测模式":
        st.header("小麦干旱检测")

        mode = st.radio(
            "选择输入方式",
            ["上传图像 (支持多张)", "手机/PC 摄像头拍摄"],
            horizontal=True,
        )

        st.sidebar.title("检测设置")
        conf_thresh = st.sidebar.slider(
            "小麦检测置信度阈值（越高越严格）",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        st.sidebar.markdown(f"**当前阈值：{conf_thresh:.2f}**")
        st.sidebar.info("置信度越高，误检越少，但可能漏检更多。")

        if mode == "上传图像 (支持多张)":
            uploaded_files = st.file_uploader(
                "上传田间图像（可多选）",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    st.subheader(f"处理图像：{uploaded_file.name}")

                    img = Image.open(uploaded_file).convert("RGB")
                    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                    with st.spinner(f"正在检测小麦（置信度 {conf_thresh:.2f}）并判断干旱..."):
                        result_bgr, log_data = process_image_with_detection(
                            frame_bgr,
                            conf_thresh=conf_thresh,
                        )

                    st.image(
                        result_bgr,
                        channels="BGR",
                        caption="检测结果（红框=干旱，绿框=正常）",
                        use_container_width=True,
                    )

                    update_log(log_data, path=uploaded_file.name)

                st.subheader("预测日志（当前会话）")
                st.dataframe(st.session_state.log_df, use_container_width=True)

                if st.button("导出日志", key="export_upload"):
                    save_log_to_disk()
                    st.success(f"日志已导出：{PREDICTION_LOG_PATH}")

        if mode == "手机/PC 摄像头拍摄":
            st.write("点击下方按钮，允许浏览器访问摄像头后拍摄。")
            camera_img = st.camera_input("拍摄小麦田间图像")

            if camera_img is not None:
                img = Image.open(camera_img).convert("RGB")
                frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                with st.spinner(f"正在检测小麦（置信度 {conf_thresh:.2f}）并判断干旱..."):
                    result_bgr, log_data = process_image_with_detection(
                        frame_bgr,
                        conf_thresh=conf_thresh,
                    )

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                photo_path = CAPTURED_DIR / f"mobile_captured_{timestamp}.png"
                cv2.imwrite(str(photo_path), result_bgr)

                st.image(
                    result_bgr,
                    channels="BGR",
                    caption=f"检测完成，已保存：{photo_path.name}",
                    use_container_width=True,
                )

                update_log(log_data, path=str(photo_path))

                st.subheader("预测日志")
                st.dataframe(st.session_state.log_df, use_container_width=True)

    if tab == "训练数据查看":
        show_training_results()

    with st.expander("全局日志管理", expanded=False):
        if st.button("清除日志", key="clear_log"):
            st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)
            save_log_to_disk()
            st.success("日志已清除")

        if st.button("导出日志", key="export_global"):
            if "log_df" in st.session_state and not st.session_state.log_df.empty:
                save_log_to_disk()
                st.success(f"日志已导出：{PREDICTION_LOG_PATH}")
            else:
                st.warning("暂无日志可导出")

    st.sidebar.title("使用说明")
    st.sidebar.info(
        "**PC端**：可上传图片或使用电脑摄像头\n\n"
        "**手机端**：浏览器打开页面后允许摄像头权限即可拍摄"
    )
    st.sidebar.write("如需公网访问，可使用：`ngrok http 8501`")


if __name__ == "__main__":
    main()