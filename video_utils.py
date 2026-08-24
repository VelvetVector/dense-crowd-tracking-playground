"""
Frame extraction + drawing helpers. Nothing fancy, just wraps cv2 calls we'd
otherwise be repeating in app.py a bunch of times.
"""

import cv2
import numpy as np

DEFAULT_RESIZE_WIDTH = 640  # keep frames smallish so playback + inference don't crawl


def extract_frames(video_path, frame_skip=1, resize_width=DEFAULT_RESIZE_WIDTH):
    """
    Reads the whole video into a list of frames (BGR numpy arrays), skipping
    every `frame_skip` frames and resizing down to resize_width.

    Loading the whole thing into memory is a little wasteful for long videos
    but keeps the playback loop dead simple (no re-seeking mid-stream) and is
    fine for the short demo clips this tool is meant for.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video file: {video_path}")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            h, w = frame.shape[:2]
            if w > resize_width:
                scale = resize_width / w
                frame = cv2.resize(frame, (resize_width, int(h * scale)))
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def draw_detections(frame, boxes_with_conf):
    """boxes_with_conf: list of [x1,y1,x2,y2,conf]. Plain boxes, no ID yet -
    used for the raw detection preview before tracking is applied."""
    out = frame.copy()
    for x1, y1, x2, y2, conf in boxes_with_conf:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 200, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(out, label, (p1[0], max(p1[1] - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    return out


def draw_tracks(frame, tracked_boxes):
    """tracked_boxes: list of [x1,y1,x2,y2,track_id]. Colors per-id so the
    same person keeps the same color as long as the tracker keeps the same
    ID - makes an ID switch visually obvious (color jumps) during playback."""
    out = frame.copy()
    for x1, y1, x2, y2, tid in tracked_boxes:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        color = _color_for_id(tid)
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"ID {tid}"
        cv2.putText(out, label, (p1[0], max(p1[1] - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def _color_for_id(track_id):
    # cheap deterministic-ish color per id, good enough to visually tell ids
    # apart, don't need anything smarter than this
    rng = np.random.RandomState(track_id * 37 + 7)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


def bgr_to_rgb(frame):
    # st.image wants RGB, cv2 gives us BGR - easy to forget this and get
    # weird colors, so it's its own function as a reminder
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
