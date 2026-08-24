"""
Dense Crowd Tracking Playground
--------------------------------
Live sandbox for messing with model size / conf threshold / tracker choice
and actually SEEING what changes, instead of just reading about it in the
assignment readme. Built on top of the crowd tracker dashboard from last
project - reusing the SORT/ByteTrack/metrics logic from there, just wired up
for live interactive playback instead of a precomputed report.

Not trying to make this pretty, just functional. Sidebar has all the knobs,
main area has the two video panels + live metrics under each.
"""

import os
import time
from datetime import datetime

import streamlit as st

from detector import load_model, run_detection, filter_by_conf, MODEL_SIZES
from trackers.sort_tracker import SortTracker
from trackers.byte_tracker import ByteTracker
from metrics import MetricsTracker
import video_utils

PROCESS_LOG_PATH = "process_log.txt"
SAMPLE_DATA_DIR = "sample_data"
DEFAULT_FRAME_SKIP = 2  # every 2nd frame - keeps playback from crawling on CPU
METRICS_REFRESH_EVERY_N_FRAMES = 3  # don't recompute metrics every single frame, a little wasteful
PLAYBACK_SLEEP_SECONDS = 0.15  # how long between frames in the fake "playback" loop

st.set_page_config(page_title="Dense Crowd Tracking Playground", layout="wide")


# ---------- process log helpers ----------

def append_log_line(line):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_line = f"[{timestamp}] {line}\n"
    with open(PROCESS_LOG_PATH, "a") as f:
        f.write(full_line)


def read_log():
    if not os.path.exists(PROCESS_LOG_PATH):
        return []
    with open(PROCESS_LOG_PATH, "r") as f:
        lines = f.readlines()
    return list(reversed(lines))  # most recent at top


# ---------- cached model loading ----------
# simple dict cache in session_state, per the "keep caching simple" note.
# st.cache_resource would also work but this is easy to reason about and we
# only ever have 5 possible models anyway.

def get_model(size_name):
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}
    if size_name not in st.session_state.model_cache:
        with st.spinner(f"loading {size_name} (first time can take a bit, downloads weights)..."):
            st.session_state.model_cache[size_name] = load_model(size_name)
    return st.session_state.model_cache[size_name]


# ---------- video source ----------

def find_sample_video():
    if not os.path.isdir(SAMPLE_DATA_DIR):
        return None
    for fname in os.listdir(SAMPLE_DATA_DIR):
        if fname.lower().endswith((".mp4", ".avi", ".mov")):
            return os.path.join(SAMPLE_DATA_DIR, fname)
    return None


def main():
    st.title("Dense Crowd Tracking Playground")
    st.caption(
        "messing around with detector size / conf threshold / tracker choice "
        "for the dense crowd tracking assignment - live version of the "
        "dashboard from last project"
    )

    # ---------------- sidebar controls ----------------
    st.sidebar.header("controls")

    uploaded_file = st.sidebar.file_uploader(
        "upload a clip", type=["mp4", "avi", "mov"]
    )

    video_path = None
    if uploaded_file is not None:
        # dump to a temp path so cv2 can open it by filename
        video_path = os.path.join("_uploaded_tmp" + os.path.splitext(uploaded_file.name)[1])
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        sample = find_sample_video()
        if sample:
            video_path = sample
            st.sidebar.info(f"no upload - using sample: {os.path.basename(sample)}")
        else:
            st.sidebar.warning(
                "no video uploaded and sample_data/ is empty. "
                "drop a clip in sample_data/ or use the uploader above."
            )

    model_size = st.sidebar.selectbox("model size (yolov8)", MODEL_SIZES, index=1)
    st.sidebar.caption(
        "bigger model = catches more people in the packed part of the frame, "
        "but noticeably slower per frame. yolov8x on CPU is painful, heads up."
    )

    conf_threshold = st.sidebar.slider(
        "confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01
    )

    frame_skip = st.sidebar.slider(
        "frame skip (higher = faster, choppier)", min_value=1, max_value=10,
        value=DEFAULT_FRAME_SKIP,
    )

    run_button = st.sidebar.button("run / restart playback")

    st.sidebar.divider()
    st.sidebar.subheader("my process")
    note_text = st.sidebar.text_input("quick note", key="note_input")
    if st.sidebar.button("add note"):
        if note_text.strip():
            append_log_line(note_text.strip())
            st.session_state.note_input = ""
            st.rerun()

    with st.sidebar.expander("process log (most recent first)"):
        for line in read_log():
            st.text(line.strip())

    # detect config changes and auto-log them, lightweight - just compares
    # to whatever was logged last time in session_state, not a full audit
    config_key = (model_size, round(conf_threshold, 2))
    if st.session_state.get("last_logged_config") != config_key:
        if "last_logged_config" in st.session_state:
            append_log_line(
                f"switched to model={model_size}, conf={conf_threshold:.2f}"
            )
        st.session_state.last_logged_config = config_key

    if not video_path:
        st.stop()

    if not run_button and "frames_cache_path" not in st.session_state:
        st.info("hit 'run / restart playback' in the sidebar to start.")
        st.stop()

    # ---------------- load + cache frames for this video ----------------
    if run_button or st.session_state.get("frames_cache_path") != (video_path, frame_skip):
        with st.spinner("extracting frames..."):
            try:
                frames = video_utils.extract_frames(video_path, frame_skip=frame_skip)
            except IOError as e:
                st.error(f"couldn't read that video: {e}")
                st.stop()
        if len(frames) == 0:
            st.error("got zero frames out of that video, is the format actually supported?")
            st.stop()
        st.session_state.frames = frames
        st.session_state.frames_cache_path = (video_path, frame_skip)
        append_log_line(
            f"loaded video, {len(frames)} frames after skip={frame_skip}, "
            f"model={model_size}, conf={conf_threshold:.2f}"
        )

    frames = st.session_state.frames
    st.caption(f"{len(frames)} frames loaded (after frame skip)")

    model = get_model(model_size)

    # ---------------- run both trackers side by side ----------------
    col_sort, col_byte = st.columns(2)

    with col_sort:
        st.subheader("SORT")
        sort_placeholder = st.empty()
        sort_metrics_placeholder = st.empty()

    with col_byte:
        st.subheader("ByteTrack")
        byte_placeholder = st.empty()
        byte_metrics_placeholder = st.empty()

    sort_tracker = SortTracker()
    byte_tracker = ByteTracker()
    sort_tracker.reset()  # zero out the shared id counter so ids start fresh each run
    byte_tracker.reset()
    sort_metrics = MetricsTracker()
    byte_metrics = MetricsTracker()

    # cache raw detections per frame so dragging the conf slider later
    # doesn't force us to rerun YOLO on every single tick - re-filter in
    # python instead, way more responsive
    if "raw_detections_cache_key" not in st.session_state or \
       st.session_state.get("raw_detections_cache_key") != (video_path, frame_skip, model_size):
        with st.spinner(f"running {model_size} detection on all frames (only happens once per model/video combo)..."):
            raw_dets = []
            progress = st.progress(0)
            for i, frame in enumerate(frames):
                raw_dets.append(run_detection(model, frame))
                if i % 5 == 0:
                    progress.progress(min(i / len(frames), 1.0))
            progress.empty()
        st.session_state.raw_detections = raw_dets
        st.session_state.raw_detections_cache_key = (video_path, frame_skip, model_size)

    raw_detections = st.session_state.raw_detections

    for i, frame in enumerate(frames):
        raw_dets_this_frame = raw_detections[i]

        # SORT: gets pre-filtered detections (that's just how basic
        # threshold-then-track works, no second chance for low-conf boxes)
        filtered_for_sort = filter_by_conf(raw_dets_this_frame, conf_threshold)
        sort_boxes_only = [d[:4] for d in filtered_for_sort]
        sort_results = sort_tracker.update(
            [d + [1.0] for d in sort_boxes_only]  # conf not used downstream by SORT, just needs the slot
        )

        # ByteTrack: gets ALL detections down to the low-conf floor, does its
        # own internal splitting by conf_threshold - this is the whole point
        byte_results = byte_tracker.update(raw_dets_this_frame, conf_threshold)

        sort_metrics.add_frame(sort_results)
        byte_metrics.add_frame(byte_results)

        sort_frame_drawn = video_utils.draw_tracks(frame, sort_results)
        byte_frame_drawn = video_utils.draw_tracks(frame, byte_results)

        sort_placeholder.image(video_utils.bgr_to_rgb(sort_frame_drawn))
        byte_placeholder.image(video_utils.bgr_to_rgb(byte_frame_drawn))

        if i % METRICS_REFRESH_EVERY_N_FRAMES == 0 or i == len(frames) - 1:
            sm = sort_metrics.compute()
            bm = byte_metrics.compute()
            with sort_metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("ID switches", sm["id_switches"])
                m2.metric("fragmentation", sm["fragmentation"])
                m3.metric("avg track len", sm["avg_track_length"])
            with byte_metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("ID switches", bm["id_switches"])
                m2.metric("fragmentation", bm["fragmentation"])
                m3.metric("avg track len", bm["avg_track_length"])

        time.sleep(PLAYBACK_SLEEP_SECONDS)

    st.success("playback done. tweak a control and hit 'run / restart playback' to go again.")


if __name__ == "__main__":
    main()
