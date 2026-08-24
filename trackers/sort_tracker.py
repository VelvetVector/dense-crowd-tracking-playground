"""
SORT tracker - lifted mostly as-is from the crowd dashboard project last sem.
Kalman filter per track (constant velocity model on the box center + scale/ratio)
+ Hungarian matching on IoU between predicted boxes and new detections.

This is the "dumb but fast" tracker - no re-id, no appearance features, just motion.
That's exactly why it loses IDs across occlusion, which is the whole point of
having it side by side with ByteTrack in the playground.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box_a, box_b):
    # boxes are [x1, y1, x2, y2]
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class KalmanBoxTracker:
    """
    One of these per tracked object. State is [cx, cy, s, r, vx, vy, vs]
    where s = area (scale), r = aspect ratio (kept constant, we don't model
    r velocity - that's how the og SORT paper does it too).
    """

    count = 0

    def __init__(self, bbox):
        # state vector
        self.x = np.zeros((7, 1))
        cx, cy, s, r = self._bbox_to_z(bbox)
        self.x[:4, 0] = [cx, cy, s, r]

        # constant velocity motion model
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0  # cx += vx, cy += vy, s += vs

        self.H = np.zeros((4, 7))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # process / measurement noise - not tuned super carefully, defaults
        # that worked ok last time. could probably do better with a grid search
        # but that's overkill for this
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 100.0  # less confident about velocity initially

        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] *= 0.1

        self.R = np.eye(4) * 1.0

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []  # bbox history for this track, used by metrics.py

    @staticmethod
    def _bbox_to_z(bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = max(w * h, 1e-6)
        r = w / max(h, 1e-6)
        return cx, cy, s, r

    @staticmethod
    def _z_to_bbox(cx, cy, s, r):
        w = np.sqrt(max(s * r, 1e-6))
        h = s / max(w, 1e-6)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return [x1, y1, x2, y2]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        cx, cy, s, r = self.x[:4, 0]
        bbox = self._z_to_bbox(cx, cy, s, r)
        self.history.append(bbox)
        return bbox

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        z = np.array(self._bbox_to_z(bbox)).reshape((4, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self):
        cx, cy, s, r = self.x[:4, 0]
        return self._z_to_bbox(cx, cy, s, r)


class SortTracker:
    """
    Wraps a bunch of KalmanBoxTracker instances + does the frame-to-frame
    association. Call update(detections) once per frame, detections is a
    list of [x1, y1, x2, y2, conf].

    Returns list of [x1, y1, x2, y2, track_id].
    """

    def __init__(self, iou_threshold=0.3, max_age=15, min_hits=1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []

    def update(self, detections):
        # predict step for every existing tracker first
        predicted_boxes = []
        for t in self.trackers:
            predicted_boxes.append(t.predict())

        det_boxes = [d[:4] for d in detections]

        matches, unmatched_dets, unmatched_trks = self._associate(
            det_boxes, predicted_boxes
        )

        for trk_idx, det_idx in matches:
            self.trackers[trk_idx].update(det_boxes[det_idx])

        # new tracker for every unmatched detection - this is exactly where
        # SORT hands out a fresh ID after an occlusion, since the old track
        # aged out and a "new" detection shows up with no motion history
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(det_boxes[det_idx]))

        # drop trackers that have gone stale too long
        alive = []
        results = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                alive.append(t)
                if t.hit_streak >= self.min_hits or t.hits >= self.min_hits:
                    bbox = t.get_state()
                    results.append(bbox + [t.id])
        self.trackers = alive
        return results

    def _associate(self, det_boxes, trk_boxes):
        if len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(trk_boxes)))

        iou_matrix = np.zeros((len(trk_boxes), len(det_boxes)), dtype=np.float32)
        for t_idx, tb in enumerate(trk_boxes):
            for d_idx, db in enumerate(det_boxes):
                iou_matrix[t_idx, d_idx] = iou(tb, db)

        # hungarian wants a cost matrix, we have a similarity matrix -> negate
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches = []
        unmatched_trks = set(range(len(trk_boxes)))
        unmatched_dets = set(range(len(det_boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                unmatched_trks.discard(r)
                unmatched_dets.discard(c)

        return matches, sorted(unmatched_dets), sorted(unmatched_trks)

    def reset(self):
        # called when the video/threshold changes and we want a clean slate
        self.trackers = []
        KalmanBoxTracker.count = 0
