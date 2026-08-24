"""
Thin wrapper around ultralytics YOLOv8. Handles loading a model by size name
and running inference on a single frame, returning raw detections down to a
low floor confidence (real filtering by the UI slider happens in app.py so
we're not reloading/rerunning YOLO every time someone nudges the slider).

Only keeping the "person" class (COCO class 0) since this is a crowd/people
tracking tool - detecting cars/dogs/whatever in the background would just be
noise for this assignment.
"""

from ultralytics import YOLO

PERSON_CLASS_ID = 0

# floor we run YOLO's own conf at - lower than any slider value the UI
# exposes so we always have the low-conf boxes on hand for ByteTrack's
# second matching stage. don't set this to 0, gets way too noisy/slow.
YOLO_INTERNAL_CONF_FLOOR = 0.05

MODEL_SIZES = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]


def load_model(size_name):
    """
    size_name like 'yolov8n'. ultralytics will auto-download the .pt weights
    on first use if they're not already cached locally - that first call can
    take a bit, this is normal (see README known limitations).
    """
    if size_name not in MODEL_SIZES:
        raise ValueError(f"unknown model size: {size_name}")
    weights_file = f"{size_name}.pt"
    model = YOLO(weights_file)
    return model


def run_detection(model, frame):
    """
    frame: a single BGR numpy image (as read by cv2).
    returns: list of [x1, y1, x2, y2, conf] for detected people, at or above
    YOLO_INTERNAL_CONF_FLOOR. Caller filters further by the UI's actual
    threshold slider.
    """
    results = model.predict(
        frame,
        conf=YOLO_INTERNAL_CONF_FLOOR,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    detections = []
    if len(results) == 0:
        return detections

    boxes = results[0].boxes
    if boxes is None:
        return detections

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])

    return detections


def filter_by_conf(detections, conf_threshold):
    # just python-side filtering, this is what the slider actually drives
    # frame to frame so we're not re-running the model on every tick
    return [d for d in detections if d[4] >= conf_threshold]
