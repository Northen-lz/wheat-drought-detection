import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.config import (
    CAPTURED_DIR,
    PREDICTION_LOG_PATH,
    TRAIN_RESULTS_DIR,
    RAW_DATA_DIR,
    DET_MODEL_PATH,
    CLS_MODEL_PATH,
)
from utils.dashboard import (
    apply_dashboard_style,
    build_log_summary,
    close_section_box,
    render_metric_card,
    render_section_box,
    render_title,
)
from utils.inference import process_image_with_detection


st.set_page_config(
    page_title="小麦干旱智能检测系统数据大屏",
    page_icon="🌾",
    layout="wide",
)

LOG_COLUMNS = ["图像", "小麦株数", "干旱株数", "干旱率", "时间戳"]


def init_log():
    if "log_df" not in st.session_state:
        if PREDICTION_LOG_PATH.exists():
            try:
                st.session_state.log_df = pd.read_csv(PREDICTION_LOG_PATH)
            except Exception:
                st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)
        else:
            st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)

    if "last_original_image" not in st.session_state:
        st.session_state.last_original_image = None
    if "last_result_image" not in st.session_state:
        st.session_state.last_result_image = None
    if "last_log_data" not in st.session_state:
        st.session_state.last_log_data = None


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


def get_training_df():
    csv_path = TRAIN_RESULTS_DIR / "results.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def get_dataset_summary():
    summary = {
        "样本总数": 0,
        "control 数量": 0,
        "drought 数量": 0,
    }

    y_path = RAW_DATA_DIR / "YWheatCW.npy"
    if y_path.exists():
        try:
            y = np.load(y_path)
            summary["样本总数"] = int(len(y))
            summary["control 数量"] = int(np.sum(y == 0))
            summary["drought 数量"] = int(np.sum(y == 1))
        except Exception:
            pass
    return summary


def get_model_summary():
    def file_mb(p: Path):
        if p.exists():
            return round(p.stat().st_size / (1024 * 1024), 2)
        return None

    return {
        "检测模型路径": str(DET_MODEL_PATH),
        "分类模型路径": str(CLS_MODEL_PATH),
        "检测模型大小MB": file_mb(DET_MODEL_PATH),
        "ONNX模型大小MB": file_mb(CLS_MODEL_PATH),
    }


def parse_rate_series(df: pd.DataFrame):
    temp = df.copy()
    if "干旱率" not in temp.columns:
        return pd.Series(dtype=float)
    temp["干旱率数值"] = (
        temp["干旱率"].astype(str).str.replace("%", "", regex=False).str.strip()
    )
    temp["干旱率数值"] = pd.to_numeric(temp["干旱率数值"], errors="coerce").fillna(0.0)
    return temp["干旱率数值"]


def render_top_metrics(log_df: pd.DataFrame, train_df: pd.DataFrame):
    summary = build_log_summary(log_df)

    acc_value = "--"
    val_loss_value = "--"

    if not train_df.empty:
        if "metrics/accuracy_top1" in train_df.columns:
            try:
                acc = float(train_df["metrics/accuracy_top1"].dropna().iloc[-1]) * 100
                acc_value = f"{acc:.2f}%"
            except Exception:
                pass

        if "val/loss" in train_df.columns:
            try:
                val_loss_value = f"{float(train_df['val/loss'].dropna().iloc[-1]):.4f}"
            except Exception:
                pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("累计检测图像数", str(summary["total_images"]), "日志中已记录图像")
    with c2:
        render_metric_card("累计识别小麦株数", str(summary["total_wheat"]), "检测框总数")
    with c3:
        render_metric_card("累计干旱株数", str(summary["total_drought"]), "分类为 drought")
    with c4:
        render_metric_card(
            "实验准确率 / 最终Val Loss",
            acc_value,
            f"Val Loss: {val_loss_value}" if val_loss_value != "--" else "未读取到训练结果",
        )


def render_single_detection_panel():
    st.markdown("### 🛰️ 单张检测")

    c1, c2 = st.columns([0.72, 0.28])
    with c1:
        uploaded_file = st.file_uploader(
            "上传待检测图像",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="single_uploader",
        )
    with c2:
        conf_thresh = st.slider(
            "置信度阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="single_conf",
        )

    camera_mode = st.checkbox("使用手机/PC 摄像头拍摄", value=False)

    if camera_mode:
        camera_img = st.camera_input("拍摄田间图像", key="single_camera")
        if camera_img is not None:
            img = Image.open(camera_img).convert("RGB")
            frame_rgb = np.array(img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            try:
                with st.spinner(f"正在检测，阈值 {conf_thresh:.2f} ..."):
                    result_bgr, log_data = process_image_with_detection(
                        frame_bgr.copy(),
                        conf_thresh=conf_thresh,
                    )
            except Exception as e:
                st.error(f"检测失败：{e}")
                return

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            photo_path = CAPTURED_DIR / f"mobile_captured_{timestamp}.png"
            cv2.imwrite(str(photo_path), result_bgr)

            st.session_state.last_original_image = frame_rgb
            st.session_state.last_result_image = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.last_log_data = log_data
            update_log(log_data, path=str(photo_path))

    elif uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        frame_rgb = np.array(img)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        try:
            with st.spinner(f"正在检测，阈值 {conf_thresh:.2f} ..."):
                result_bgr, log_data = process_image_with_detection(
                    frame_bgr.copy(),
                    conf_thresh=conf_thresh,
                )
        except Exception as e:
            st.error(f"检测失败：{e}")
            return

        st.session_state.last_original_image = frame_rgb
        st.session_state.last_result_image = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        st.session_state.last_log_data = log_data
        update_log(log_data, path=uploaded_file.name)

    col1, col2, col3 = st.columns([1.05, 1.05, 0.85])

    with col1:
        render_section_box("原始图像")
        if st.session_state.last_original_image is not None:
            st.image(st.session_state.last_original_image, use_container_width=True)
        else:
            st.info("请上传图像或使用摄像头拍摄。")
        close_section_box()

    with col2:
        render_section_box("检测结果图")
        if st.session_state.last_result_image is not None:
            st.image(st.session_state.last_result_image, use_container_width=True)
        else:
            st.info("检测结果将在这里显示。")
        close_section_box()

    with col3:
        render_section_box("当前检测摘要")
        log_data = st.session_state.last_log_data
        if log_data:
            st.metric("图像名称", str(log_data.get("图像", "uploaded.png")))
            st.metric("小麦株数", int(log_data.get("小麦株数", 0)))
            st.metric("干旱株数", int(log_data.get("干旱株数", 0)))
            st.metric("干旱率", str(log_data.get("干旱率", "0%")))
            st.caption(f"检测时间：{log_data.get('时间戳', '')}")
        else:
            st.info("等待检测结果。")
        close_section_box()


def render_batch_detection_panel():
    st.markdown("### 📦 批量检查")

    c1, c2, c3 = st.columns([0.58, 0.22, 0.20])
    with c1:
        uploaded_files = st.file_uploader(
            "批量上传待检测图像",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="batch_uploader",
        )
    with c2:
        conf_thresh = st.slider(
            "批量阈值",
            0.0,
            1.0,
            0.5,
            0.05,
            key="batch_conf",
        )
    with c3:
        preview_limit = st.number_input(
            "预览前N张",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="preview_limit",
        )

    if not uploaded_files:
        st.info("请上传多张图片进行批量检查。")
        return

    if st.button("开始批量检查", key="run_batch"):
        batch_rows = []
        preview_results = []

        progress = st.progress(0)
        status = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            status.write(f"正在处理：{uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")

            img = Image.open(uploaded_file).convert("RGB")
            frame_rgb = np.array(img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            try:
                result_bgr, log_data = process_image_with_detection(
                    frame_bgr.copy(),
                    conf_thresh=conf_thresh,
                )
            except Exception as e:
                log_data = {
                    "图像": uploaded_file.name,
                    "小麦株数": 0,
                    "干旱株数": 0,
                    "干旱率": "0%",
                    "时间戳": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                batch_rows.append(
                    {
                        "图像": uploaded_file.name,
                        "状态": f"失败: {e}",
                        "小麦株数": 0,
                        "干旱株数": 0,
                        "干旱率": "0%",
                    }
                )
                progress.progress((idx + 1) / len(uploaded_files))
                continue

            update_log(log_data, path=uploaded_file.name)

            batch_rows.append(
                {
                    "图像": uploaded_file.name,
                    "状态": "成功",
                    "小麦株数": log_data["小麦株数"],
                    "干旱株数": log_data["干旱株数"],
                    "干旱率": log_data["干旱率"],
                }
            )

            if len(preview_results) < preview_limit:
                preview_results.append(
                    {
                        "图像": uploaded_file.name,
                        "原图": frame_rgb,
                        "结果图": cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB),
                        "摘要": log_data,
                    }
                )

            progress.progress((idx + 1) / len(uploaded_files))

        status.success("批量检查完成")

        batch_df = pd.DataFrame(batch_rows)

        left, right = st.columns([0.9, 1.1])
        with left:
            render_section_box("批量结果汇总")
            st.dataframe(batch_df, use_container_width=True, height=360)
            close_section_box()

        with right:
            render_section_box("批量统计")
            if not batch_df.empty:
                ok_df = batch_df[batch_df["状态"] == "成功"].copy()
                if not ok_df.empty:
                    ok_df["小麦株数"] = pd.to_numeric(ok_df["小麦株数"], errors="coerce").fillna(0)
                    ok_df["干旱株数"] = pd.to_numeric(ok_df["干旱株数"], errors="coerce").fillna(0)
                    ok_df["干旱率数值"] = (
                        ok_df["干旱率"].astype(str).str.replace("%", "", regex=False)
                    )
                    ok_df["干旱率数值"] = pd.to_numeric(ok_df["干旱率数值"], errors="coerce").fillna(0)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("成功图像数", int(len(ok_df)))
                    c2.metric("总小麦株数", int(ok_df["小麦株数"].sum()))
                    c3.metric("平均干旱率", f"{ok_df['干旱率数值'].mean():.2f}%")

                    st.bar_chart(ok_df[["小麦株数", "干旱株数"]], use_container_width=True)
                else:
                    st.warning("批量检查均失败。")
            close_section_box()

        render_section_box("批量预览（已限制显示数量，避免页面过长）")
        if preview_results:
            for item in preview_results:
                with st.expander(f"查看：{item['图像']}"):
                    c1, c2, c3 = st.columns([1, 1, 0.8])
                    with c1:
                        st.image(item["原图"], caption="原图", use_container_width=True)
                    with c2:
                        st.image(item["结果图"], caption="结果图", use_container_width=True)
                    with c3:
                        st.write(item["摘要"])
        else:
            st.info("暂无预览结果。")
        close_section_box()


def render_experiment_panel(train_df: pd.DataFrame):
    st.markdown("### 🧪 实验数据与模型表现")

    results_png_path = TRAIN_RESULTS_DIR / "results.png"
    confusion_path = TRAIN_RESULTS_DIR / "confusion_matrix.png"
    confusion_norm_path = TRAIN_RESULTS_DIR / "confusion_matrix_normalized.png"

    left, right = st.columns([1.15, 0.85])

    with left:
        render_section_box("训练曲线与实验图像")
        tabs = st.tabs(["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"])

        with tabs[0]:
            if results_png_path.exists():
                st.image(str(results_png_path), use_container_width=True)
            else:
                st.warning("未找到 results.png")

        with tabs[1]:
            if confusion_path.exists():
                st.image(str(confusion_path), use_container_width=True)
            else:
                st.warning("未找到 confusion_matrix.png")

        with tabs[2]:
            if confusion_norm_path.exists():
                st.image(str(confusion_norm_path), use_container_width=True)
            else:
                st.warning("未找到 confusion_matrix_normalized.png")
        close_section_box()

    with right:
        render_section_box("实验结果摘要")
        if train_df.empty:
            st.warning("未读取到 results.csv")
        else:
            latest = train_df.tail(1).iloc[0]

            if "train/loss" in train_df.columns:
                st.metric("最终 Train Loss", f"{float(latest['train/loss']):.4f}")
            if "val/loss" in train_df.columns:
                st.metric("最终 Val Loss", f"{float(latest['val/loss']):.4f}")
            if "metrics/accuracy_top1" in train_df.columns:
                st.metric("Top-1 Accuracy", f"{float(latest['metrics/accuracy_top1']) * 100:.2f}%")

            st.markdown("#### 最近 10 条训练记录")
            st.dataframe(train_df.tail(10), use_container_width=True, height=280)
        close_section_box()

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        render_section_box("Loss 曲线")
        if not train_df.empty:
            loss_cols = [c for c in ["train/loss", "val/loss"] if c in train_df.columns]
            if loss_cols:
                st.line_chart(train_df[loss_cols], use_container_width=True)
            else:
                st.info("results.csv 中未找到 loss 列。")
        else:
            st.info("暂无训练数据。")
        close_section_box()

    with bottom_right:
        render_section_box("Accuracy 曲线")
        if not train_df.empty and "metrics/accuracy_top1" in train_df.columns:
            st.line_chart(train_df["metrics/accuracy_top1"], use_container_width=True)
        else:
            st.info("results.csv 中未找到 metrics/accuracy_top1。")
        close_section_box()


def render_achievement_panel(log_df: pd.DataFrame, train_df: pd.DataFrame):
    st.markdown("### 🏆 项目成果整合")

    dataset_summary = get_dataset_summary()
    model_summary = get_model_summary()
    log_summary = build_log_summary(log_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据集样本总数", dataset_summary["样本总数"])
    c2.metric("control 数量", dataset_summary["control 数量"])
    c3.metric("drought 数量", dataset_summary["drought 数量"])
    c4.metric("累计检测图像数", log_summary["total_images"])

    left, right = st.columns([0.95, 1.05])

    with left:
        render_section_box("模型部署成果")
        st.write(f"检测模型：`{model_summary['检测模型路径']}`")
        st.write(f"分类模型：`{model_summary['分类模型路径']}`")
        st.write(f"检测模型大小：`{model_summary['检测模型大小MB']} MB`")
        st.write(f"ONNX 模型大小：`{model_summary['ONNX模型大小MB']} MB`")
        close_section_box()

    with right:
        render_section_box("应用成果摘要")
        st.write("• 已实现单张检测、批量检查、实验结果可视化、历史日志追踪")
        st.write("• 已集成目标检测 + ONNX 分类推理流程")
        st.write("• 可展示训练曲线、混淆矩阵、准确率、干旱率趋势")
        st.write("• 支持检测日志导出，便于实验记录和答辩演示")
        close_section_box()

    low, high = st.columns(2)

    with low:
        render_section_box("最近干旱率趋势")
        if not log_df.empty:
            rate_series = parse_rate_series(log_df)
            st.line_chart(rate_series, use_container_width=True)
        else:
            st.info("暂无日志趋势数据。")
        close_section_box()

    with high:
        render_section_box("训练指标概览")
        if not train_df.empty:
            cols = [c for c in ["train/loss", "val/loss", "metrics/accuracy_top1"] if c in train_df.columns]
            if cols:
                st.dataframe(train_df[cols].tail(10), use_container_width=True, height=260)
            else:
                st.info("训练表中缺少核心指标列。")
        else:
            st.info("暂无训练实验数据。")
        close_section_box()


def render_history_panel(log_df: pd.DataFrame):
    st.markdown("### 📚 历史记录")

    left, right = st.columns([1.1, 0.9])

    with left:
        render_section_box("最近检测记录")
        if log_df.empty:
            st.info("暂无日志。")
        else:
            display_df = log_df.copy().tail(20).iloc[::-1]
            st.dataframe(display_df, use_container_width=True, height=420)
        close_section_box()

    with right:
        render_section_box("干旱率趋势")
        if log_df.empty:
            st.info("暂无日志。")
        else:
            rate_series = parse_rate_series(log_df)
            st.line_chart(rate_series, use_container_width=True)
        close_section_box()


def main():
    apply_dashboard_style()
    init_log()

    log_df = st.session_state.log_df
    train_df = get_training_df()

    render_title()

    top_left, top_right = st.columns([0.82, 0.18])
    with top_left:
        st.caption("当前时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with top_right:
        st.caption("模型状态：在线")

    render_top_metrics(log_df, train_df)

    tabs = st.tabs(["单张检测", "批量检查", "实验数据", "项目成果", "历史记录"])

    with tabs[0]:
        render_single_detection_panel()

    with tabs[1]:
        render_batch_detection_panel()

    with tabs[2]:
        render_experiment_panel(train_df)

    with tabs[3]:
        render_achievement_panel(st.session_state.log_df, train_df)

    with tabs[4]:
        render_history_panel(st.session_state.log_df)

    with st.sidebar:
        st.title("系统控制")
        st.info("本页面支持单张检测、批量检查、实验数据可视化、项目成果展示和历史日志分析。")

        if st.button("导出日志"):
            save_log_to_disk()
            st.success(f"日志已导出：{PREDICTION_LOG_PATH}")

        if st.button("清空日志"):
            st.session_state.log_df = pd.DataFrame(columns=LOG_COLUMNS)
            save_log_to_disk()
            st.success("日志已清空")

        st.markdown("---")
        st.caption("建议将实验图放入 outputs/train_results/")
        st.caption("支持：results.csv / results.png / confusion_matrix.png / confusion_matrix_normalized.png")


if __name__ == "__main__":
    main()