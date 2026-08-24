"""
ByteTrack - simplified. Didn't pull in the full official repo since it's a
whole separate package with its own matching/kalman code and honestly
overkill for a demo tool. This is the "spirit of ByteTrack" instead:

Core idea (this is the whole trick): don't throw away low-confidence boxes
before tracking. Most trackers filter by conf threshold BEFORE tracking, which
means an occluded person who drops to a low-confidence half-visible box just
gets deleted, then reappears later as a "new" detection with a new ID.
ByteTrack instead:
  1. splits detections into high-conf and low-conf (below the main threshold
     but above a floor) buckets
  2. matches high-conf boxes to existing tracks first (like normal IoU
     tracking)
  3. THEN tries to match remaining unmatched tracks against the low-conf
     leftover boxes - this is what "rescues" a track through a dip in
     detection confidence during occlusion instead of killing it

We reuse the same Kalman box model + IoU + Hungarian matching as sort_tracker
(no reason to write a second Kalman filter) and just add the two-stage
association on top. That's really the only difference from SORT here.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.sort_tracker import KalmanBoxTracker, iou


LOW_CONF_FLOOR = 0.1  # boxes below this never even get considered, pure noise


class ByteTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=1):
        # note: max_age is higher than SORT's default on purpose - the whole
        # point is being more patient about keeping a track alive through a
        # rough patch, that's what actually prevents the ID switch
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []

    def update(self, detections, conf_threshold):
        """
        detections: list of [x1, y1, x2, y2, conf] - should include the low
        conf ones too (down to LOW_CONF_FLOOR), NOT pre-filtered by the UI
        slider. We do the splitting ourselves in here.
        conf_threshold: the "main" threshold from the UI slider - used to
        split into high/low buckets, this is the whole ByteTrack trick.
        """
        predicted_boxes = [t.predict() for t in self.trackers]

        high = [d for d in detections if d[4] >= conf_threshold]
        low = [d for d in detections if LOW_CONF_FLOOR <= d[4] < conf_threshold]

        high_boxes = [d[:4] for d in high]
        low_boxes = [d[:4] for d in low]

        # stage 1: match high-conf detections against all tracks
        matches1, unmatched_dets1, unmatched_trks1 = self._associate(
            high_boxes, predicted_boxes
        )
        for trk_idx, det_idx in matches1:
            self.trackers[trk_idx].update(high_boxes[det_idx])

        # stage 2: leftover tracks (didn't find a high-conf match) get a
        # second shot against the low-conf boxes. this is the rescue step.
        remaining_trk_boxes = [predicted_boxes[i] for i in unmatched_trks1]
        matches2, unmatched_low, still_unmatched_local = self._associate(
            low_boxes, remaining_trk_boxes
        )
        # remap local indices back to the real tracker list indices
        for local_trk_idx, det_idx in matches2:
            real_trk_idx = unmatched_trks1[local_trk_idx]
            self.trackers[real_trk_idx].update(low_boxes[det_idx])

        still_unmatched_trks = [unmatched_trks1[i] for i in still_unmatched_local]

        # only totally unmatched HIGH conf detections spawn new tracks -
        # low conf leftovers never start a new track, too noisy/unreliable
        # for that (this matches the actual ByteTrack paper's reasoning)
        for det_idx in unmatched_dets1:
            self.trackers.append(KalmanBoxTracker(high_boxes[det_idx]))

        alive = []
        results = []
        for i, t in enumerate(self.trackers):
            if t.time_since_update <= self.max_age:
                alive.append(t)
                if t.hit_streak >= self.min_hits or t.hits >= self.min_hits:
                    bbox = t.get_state()
                    results.append(bbox + [t.id])
        self.trackers = alive
        return results

    def _associate(self, det_boxes, trk_boxes):
        # same hungarian/IoU logic as SORT, copy-pasted rather than shared
        # because I didn't want to refactor sort_tracker.py's associate into
        # a standalone util mid-assignment and risk breaking that one - can
        # clean this dup up later if there's time
        if len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(trk_boxes)))

        iou_matrix = np.zeros((len(trk_boxes), len(det_boxes)), dtype=np.float32)
        for t_idx, tb in enumerate(trk_boxes):
            for d_idx, db in enumerate(det_boxes):
                iou_matrix[t_idx, d_idx] = iou(tb, db)

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
        self.trackers = []
        # NOTE: shares the KalmanBoxTracker.count class var with SORT, so if
        # both are running side by side (which they are, in app.py) IDs are
        # global across both panels. Fine for this demo, just means ID
        # numbers won't both start at 1 - not a bug, just how it shook out.
