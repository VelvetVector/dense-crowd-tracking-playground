"""
Tracking quality metrics, from-scratch (no external MOT eval library) same as
the earlier dashboard project. Kept simple on purpose - these aren't the
"official" MOTChallenge definitions exactly (no ground truth matching, ghost
detections, that whole apparatus) since we don't have ground-truth annotations
for whatever clip gets uploaded here. This is a rough/practical proxy version
that's good enough to *see the trend* live during playback, which is the
point of this tool.

We track, frame by frame, which track_id occupies roughly the same spatial
"slot" as before by keeping a simple history of (frame_num, track_id, bbox)
tuples per playback session, then derive:

- ID switches: a rough proxy - if a NEW track_id shows up spatially very
  close to where a track that just disappeared last was, count it as a
  switch instead of a brand new person. This is exactly the "person walks
  behind another person and comes out with a different number" case.
- Fragmentation: how many times, across the whole video, a real track breaks
  into more than one ID-labeled segment (a track "restarting" counts as a
  fragment).
- Avg track length: mean number of frames each unique track ID persisted for.
"""

import numpy as np


IOU_MATCH_FOR_SWITCH = 0.3  # spatial overlap threshold to call it "the same slot"
GAP_TOLERANCE_FRAMES = 5    # how many frames a gap can be and still count as a switch not a new person


def _iou(box_a, box_b):
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


class MetricsTracker:
    """
    Feed this frame_results (list of [x1,y1,x2,y2,track_id]) one frame at a
    time via add_frame(). Call compute() any time to get the current running
    numbers - meant to be called every few frames during playback so the
    numbers feel "live" rather than a single end-of-run dump.
    """

    def __init__(self):
        self.frame_num = 0
        # per track_id: list of frame numbers it appeared in (for avg length
        # + fragmentation - a track_id with a gap bigger than tolerance in
        # its frame list counts as fragmented)
        self.track_frames = {}
        # last known bbox + frame for every track_id we've seen, used to spot
        # a "new" id popping up in the same spot an old one just vacated
        self.last_seen = {}  # track_id -> (frame_num, bbox)
        self.id_switch_count = 0

    def reset(self):
        self.__init__()

    def add_frame(self, frame_results):
        """frame_results: list of [x1,y1,x2,y2,track_id] for this frame."""
        current_ids = set()
        for x1, y1, x2, y2, tid in frame_results:
            bbox = [x1, y1, x2, y2]
            current_ids.add(tid)

            self.track_frames.setdefault(tid, []).append(self.frame_num)

            # an ID switch = a detection gets assigned a new track ID that
            # occupies roughly the same spot a DIFFERENT, now-vanished track
            # id was last seen in, within a small frame gap - this is our
            # proxy for "same physical person, tracker lost the thread"
            if tid not in self.last_seen:
                for other_id, (last_frame, last_bbox) in self.last_seen.items():
                    if other_id == tid:
                        continue
                    gap = self.frame_num - last_frame
                    if 0 < gap <= GAP_TOLERANCE_FRAMES:
                        if _iou(bbox, last_bbox) >= IOU_MATCH_FOR_SWITCH:
                            self.id_switch_count += 1
                            break

            self.last_seen[tid] = (self.frame_num, bbox)

        self.frame_num += 1

    def compute(self):
        """
        Returns dict with id_switches, fragmentation, avg_track_length.
        Safe to call mid-playback, just reflects whatever's been fed so far.
        """
        if not self.track_frames:
            return {
                "id_switches": self.id_switch_count,
                "fragmentation": 0,
                "avg_track_length": 0.0,
            }

        fragment_count = 0
        lengths = []
        for tid, frames in self.track_frames.items():
            lengths.append(len(frames))
            frames_sorted = sorted(frames)
            # fragmentation: count gaps inside a single id's own frame list -
            # each gap bigger than tolerance = the track effectively broke
            # and picked back up (or a stale/re-used id came back), so it's
            # fragmented into multiple pieces
            gaps = 0
            for i in range(1, len(frames_sorted)):
                if frames_sorted[i] - frames_sorted[i - 1] > GAP_TOLERANCE_FRAMES:
                    gaps += 1
            if gaps > 0:
                fragment_count += gaps

        avg_length = float(np.mean(lengths)) if lengths else 0.0

        return {
            "id_switches": self.id_switch_count,
            "fragmentation": fragment_count,
            "avg_track_length": round(avg_length, 1),
        }
