# estimation_app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import tempfile
import time
import os
from collections import deque
from streamlit.runtime.media_file_storage import MediaFileStorage
import atexit

# Initialize and clear media files
def clear_temp_files():
    try:
        MediaFileStorage.clear_all()
    except:
        pass

atexit.register(clear_temp_files)
clear_temp_files()

# Suppress MediaPipe/TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------------
# App Configuration
# -------------------------
st.set_page_config(layout="wide", page_title="Pose + Segmentation + 3D Export + History")
st.title("WebRTC / Video Pose Estimation + Segmentation + 3D Export + History Buffer")

# -------------------------
# Sidebar controls
# -------------------------
model_complexity = st.sidebar.selectbox("Model Complexity", [0, 1, 2], index=1)
detection_conf = st.sidebar.slider("Detection confidence", 0.0, 1.0, 0.5, 0.05)
tracking_conf = st.sidebar.slider("Tracking confidence", 0.0, 1.0, 0.5, 0.05)
segmentation_conf = st.sidebar.slider("Segmentation threshold", 0.0, 1.0, 0.5, 0.05)
show_seg = st.sidebar.checkbox("Enable segmentation overlay", value=True)
show_3d = st.sidebar.checkbox("Show 3D plot", value=True)
update_3d_every_n_frames = st.sidebar.slider("Update 3D every N frames", 1, 30, 6)

# Buffer & Recording Controls
buffer_size = st.sidebar.slider("Landmark buffer size (frames)", 10, 500, 50, 10)
save_html_sequence = st.sidebar.checkbox("Save sequence of HTML 3D plots", value=False)
continuous_html_recording = st.sidebar.checkbox("Continuous HTML recording (single file)", value=False)

# Upload video option
video_file = st.sidebar.file_uploader("Upload video file", type=["mp4", "avi", "mov"])

# -------------------------
# MediaPipe setup
# -------------------------
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def landmarks_to_dataframe(landmarks, frame_index=None):
    rows = []
    for idx, lm in enumerate(landmarks.landmark):
        row = {
            "frame": frame_index,
            "index": idx,
            "name": mp_pose.PoseLandmark(idx).name,
            "x_norm": lm.x,
            "y_norm": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        }
        rows.append(row)
    return pd.DataFrame(rows)

def build_3d_figure(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df["x_norm"], y=-df["y_norm"], z=-df["z"],
        mode="markers+text", text=df["name"],
        marker=dict(size=4), textposition="top center",
    ))
    for conn in mp_pose.POSE_CONNECTIONS:
        a_idx = conn[0].value if hasattr(conn[0], "value") else conn[0]
        b_idx = conn[1].value if hasattr(conn[1], "value") else conn[1]
        if a_idx in df.index and b_idx in df.index:
            fig.add_trace(go.Scatter3d(
                x=[df.loc[a_idx, "x_norm"], df.loc[b_idx, "x_norm"]],
                y=[-df.loc[a_idx, "y_norm"], -df.loc[b_idx, "y_norm"]],
                z=[-df.loc[a_idx, "z"], -df.loc[b_idx, "z"]],
                mode="lines", line=dict(width=2), showlegend=False
            ))
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="auto"),
                      margin=dict(l=0, r=0, b=0, t=0))
    return fig

def apply_segmentation_mask(frame_bgr, seg_mask, seg_threshold=0.5, alpha=0.7):
    mask = (seg_mask > seg_threshold).astype(np.uint8)
    mask_3 = np.stack([mask]*3, axis=-1)
    overlay = frame_bgr.copy()
    color = np.full_like(frame_bgr, (0, 200, 0))
    overlay[mask_3 == 1] = cv2.addWeighted(frame_bgr, 0.4, color, 0.6, 0)[mask_3 == 1]
    bg = cv2.GaussianBlur(frame_bgr, (25, 25), 0)
    out = np.where(mask_3 == 1, overlay, bg)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, out, alpha, 0)

# -------------------------
# Video Processor Class
# -------------------------
class PoseVideoProcessor:
    def __init__(self):
        self.pose = None
        self.seg = None
        self.initialize_models()
        self.latest_df = None
        self.history = deque(maxlen=buffer_size)
        self.frame_counter = 0
        self.last_html_update = 0
        
    def initialize_models(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=True,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.seg = mp_selfie.SelfieSegmentation(model_selection=1)

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_counter += 1
            
            # Process pose and segmentation
            res_pose = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            seg_mask = self.seg.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).segmentation_mask if show_seg else None

            annotated = img.copy()
            
            if res_pose.pose_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    annotated, res_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
                
                # Store landmarks
                df = landmarks_to_dataframe(res_pose.pose_landmarks, frame_index=self.frame_counter)
                self.latest_df = df.set_index("index")
                self.history.append(df)
                
                # Handle HTML exports (throttled)
                current_time = time.time()
                if (save_html_sequence or continuous_html_recording) and (current_time - self.last_html_update > 1.0):
                    if save_html_sequence:
                        build_3d_figure(df).write_html(f"pose_frame_{self.frame_counter}.html")
                    if continuous_html_recording:
                        combined_df = pd.concat(self.history).reset_index(drop=True)
                        build_3d_figure(combined_df).write_html("pose_continuous.html")
                    self.last_html_update = current_time

            # Apply segmentation if enabled
            if show_seg and seg_mask is not None:
                annotated = apply_segmentation_mask(annotated, seg_mask, seg_threshold=segmentation_conf)

            return annotated
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return frame.to_ndarray(format="bgr24")

    def __del__(self):
        if self.pose:
            self.pose.close()
        if self.seg:
            self.seg.close()

# -------------------------
# Main App Logic
# -------------------------
if video_file:
    # Video file processing mode
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(video_file.read())
        temp_path = tfile.name
    
    try:
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 0

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=True,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        seg = mp_selfie.SelfieSegmentation(model_selection=1)

        history = []
        frame_counter = 0
        stframe = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.markdown(f"**Video Info:** {total_frames} frames @ {fps_video:.1f} FPS")

        show_realtime_3d = st.sidebar.checkbox("Show 3D preview while processing", value=False)
        save_annotated = st.sidebar.checkbox("Save annotated/output video", value=False)
        
        # Video writer setup
        video_writer = None
        if save_annotated:
            output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_counter += 1
            res_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res_seg = seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if show_seg else None

            # Process landmarks
            if res_pose.pose_landmarks:
                df = landmarks_to_dataframe(res_pose.pose_landmarks, frame_index=frame_counter)
                history.append(df)
                
                if show_realtime_3d and frame_counter % update_3d_every_n_frames == 0:
                    fig = build_3d_figure(df.set_index("index").reset_index())
                    st.plotly_chart(fig, use_container_width=True, height=400)

            # Annotate frame
            annotated_frame = frame.copy()
            if res_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    res_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )
            if show_seg and res_seg and res_seg.segmentation_mask is not None:
                annotated_frame = apply_segmentation_mask(
                    annotated_frame, 
                    res_seg.segmentation_mask, 
                    seg_threshold=segmentation_conf
                )

            # Initialize video writer if needed
            if save_annotated and video_writer is None:
                h, w = annotated_frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    output_path, 
                    fourcc, 
                    fps_video if fps_video > 0 else 30.0, 
                    (w, h))
            
            if save_annotated and video_writer is not None:
                video_writer.write(annotated_frame)

            # Update UI
            stframe.image(annotated_frame, channels="BGR", caption=f"Frame {frame_counter}")
            progress_bar.progress(frame_counter / total_frames)
            elapsed_time = time.time() - start_time
            status_text.text(f"Processed {frame_counter}/{total_frames} frames | {frame_counter/elapsed_time:.2f} FPS")

        # Cleanup
        cap.release()
        pose.close()
        seg.close()
        if video_writer is not None:
            video_writer.release()
        progress_bar.empty()
        status_text.empty()

        # Results display
        if history:
            full_history = pd.concat(history).reset_index(drop=True)
            st.subheader(f"Processed Video ({len(history)} frames)")
            
            # CSV Download
            st.download_button(
                "Download Video Pose CSV",
                full_history.to_csv(index=False),
                "pose_video_history.csv",
                "text/csv"
            )

            # 3D Visualization
            if show_3d:
                last_df = history[-1]
                fig = build_3d_figure(last_df.set_index("index"))
                st.plotly_chart(fig, use_container_width=True, height=480)
                
                # HTML Download
                st.download_button(
                    "Download Video 3D HTML",
                    fig.to_html(full_html=True, include_plotlyjs='cdn'),
                    "pose_video_3d.html",
                    "text/html"
                )

            # Annotated Video Download
            if save_annotated and os.path.exists(output_path):
                st.success("Annotated video saved.")
                st.video(output_path)
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Annotated Video",
                        data=f.read(),
                        file_name="annotated_pose.mp4",
                        mime="video/mp4"
                    )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if save_annotated and os.path.exists(output_path):
            os.unlink(output_path)

else:
    # Live webcam mode
    webrtc_ctx = webrtc_streamer(
        key="pose-webcam",
        video_processor_factory=PoseVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if webrtc_ctx.video_processor:
        processor = webrtc_ctx.video_processor
        if processor.latest_df is not None:
            st.subheader("Latest Frame Landmarks")
            st.dataframe(processor.latest_df.reset_index(), use_container_width=True)
            
            st.download_button(
                "Download Latest Frame CSV",
                processor.latest_df.reset_index().to_csv(index=False),
                "pose_latest.csv",
                "text/csv"
            )

            if len(processor.history) > 0:
                full_history = pd.concat(processor.history).reset_index(drop=True)
                st.subheader(f"History Buffer ({len(processor.history)} frames)")
                
                st.download_button(
                    "Download History CSV",
                    full_history.to_csv(index=False),
                    "pose_history.csv",
                    "text/csv"
                )

                if show_3d:
                    fig = build_3d_figure(processor.latest_df.reset_index())
                    st.plotly_chart(fig, use_container_width=True, height=480)
                    
                    st.download_button(
                        "Download 3D Plot HTML",
                        fig.to_html(full_html=True, include_plotlyjs='cdn'),
                        "pose_3d_latest.html",
                        "text/html"
                    )