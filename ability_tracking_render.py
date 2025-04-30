# Render 対応の ability_tracking.py コードは分割が必要なため、段階的に生成されます。
import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import subprocess

def convert_to_h264(input_path, output_path):
    command = ["ffmpeg", "-y", "-i", input_path,
               "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path]
    try:
        subprocess.run(command, check=True)
    except Exception as e:
        st.error(f"ffmpeg 実行エラー: {e}")

st.title("DBMSトラッキング（Render対応版）")

uploaded_video = st.file_uploader("動画をアップロード", type=["mp4", "avi", "mov", "mkv"])
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if "slider_frame" not in st.session_state:
        st.session_state.slider_frame = 0
    if "start_frame" not in st.session_state:
        st.session_state.start_frame = 0
    if "end_frame" not in st.session_state:
        st.session_state.end_frame = total_frames - 1

    st.sidebar.write(f"総フレーム数: {total_frames}")
    st.sidebar.write(f"FPS: {fps:.2f}")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 6, 1, 1])
    with col1:
        if st.button("⏪"):
            st.session_state.slider_frame = max(0, st.session_state.slider_frame - 10)
    with col2:
        if st.button("←"):
            st.session_state.slider_frame = max(0, st.session_state.slider_frame - 1)
    with col3:
        st.session_state.slider_frame = st.slider("現在のフレーム", 0, total_frames - 1, st.session_state.slider_frame)
    with col4:
        if st.button("→"):
            st.session_state.slider_frame = min(total_frames - 1, st.session_state.slider_frame + 1)
    with col5:
        if st.button("⏩"):
            st.session_state.slider_frame = min(total_frames - 1, st.session_state.slider_frame + 10)

    col6, col7 = st.columns(2)
    with col6:
        if st.button("この位置を初期位置に設定"):
            st.session_state.start_frame = st.session_state.slider_frame
    with col7:
        if st.button("この位置を終了位置に設定"):
            st.session_state.end_frame = st.session_state.slider_frame

    st.write(f"🔵 開始フレーム: {st.session_state.start_frame}, 🔴 終了フレーム: {st.session_state.end_frame}")

    cap = cv2.VideoCapture(video_path)
    selected_frame = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.slider_frame)
    ret, frame = cap.read()
    selected_frame = frame.copy() if ret else None
    cap.release()

    if selected_frame is not None:
        pil_image = Image.fromarray(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB))
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=pil_image,
            update_streamlit=True,
            height=selected_frame.shape[0],
            width=selected_frame.shape[1],
            drawing_mode="rect",
            key="canvas"
        )


        if st.button("▶️ トラッキングを開始") and canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            rect = canvas_result.json_data["objects"][0]
            x, y = int(rect["left"]), int(rect["top"])
            w, h = int(rect["width"]), int(rect["height"])
            init_box = (x, y, w, h)

            cap = cv2.VideoCapture(video_path)
            tracker = cv2.TrackerKCF_create()
            for i in range(st.session_state.start_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
            tracker.init(frame, init_box)

            results = []
            temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(temp_raw.name, fourcc, fps, (frame.shape[1], frame.shape[0]))

            cap = cv2.VideoCapture(video_path)
            for i in range(st.session_state.start_frame):
                cap.read()

            progress_bar = st.progress(0)
            status_text = st.empty()

            current = st.session_state.start_frame
            while current <= st.session_state.end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    cx, cy = x + w // 2, y + h // 2
                    cx = np.clip(cx, 0, frame.shape[1] - 1)
                    cy = np.clip(cy, 0, frame.shape[0] - 1)
                    b, g, r = frame[cy, cx]
                    y_val = 0.299 * r + 0.587 * g + 0.114 * b
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                else:
                    cx, cy, y_val = np.nan, np.nan, np.nan

                writer.write(frame)
                results.append({"frame": current, "x": cx, "y": cy, "Y": y_val})
                progress_bar.progress((current - st.session_state.start_frame + 1) / (st.session_state.end_frame - st.session_state.start_frame + 1))
                status_text.text(f"{current} / {st.session_state.end_frame} 処理中")
                current += 1

            cap.release()
            writer.release()
            progress_bar.empty()
            status_text.text("✅ トラッキング完了")

            df = pd.DataFrame(results)
            st.dataframe(df.head())
            st.download_button("📄 CSVダウンロード", df.to_csv(index=False).encode("utf-8-sig"), file_name="tracking_results.csv")

            st.markdown("### 📈 輝度グラフ")
            plt.figure(figsize=(8, 4))
            plt.plot(df["frame"], df["Y"])
            plt.xlabel("Frame")
            plt.ylabel("Y（輝度）")
            plt.grid(True)
            st.pyplot(plt)

            mp4_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            convert_to_h264(temp_raw.name, mp4_output)
            with open(mp4_output, "rb") as f2:
                video_bytes = f2.read()
                st.video(video_bytes)
                st.download_button("📥 トラッキング動画DL", video_bytes, file_name="tracking_output.mp4", mime="video/mp4")
