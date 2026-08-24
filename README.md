# Dense Crowd Tracking Playground

Quick interactive tool I built while working through the dense crowd tracking
assignment. The assignment readme wants us to show we understand how model
size affects recall in dense crowds, how the confidence threshold trades off
false positives vs false negatives, how tracker choice (SORT vs ByteTrack)
affects ID stability through occlusion, and that tracking quality can
actually be measured (ID switches, fragmentation, avg track length). Reading
about all that is one thing, but I wanted to actually *see* it happen live
while dragging sliders around, so this is basically a live sandbox version of
the crowd tracker dashboard from the earlier project, reusing the same SORT
(from-scratch Kalman filter + Hungarian matching) and ByteTrack logic, just
wired up for interactive frame-by-frame playback instead of a precomputed
report.

## Setup & run locally

```bash
# from inside the crowd_tracking_playground/ folder
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

You'll need a video clip to actually use it - either drop an `.mp4`/`.avi`/
`.mov` file into `sample_data/`, or just upload one through the file
uploader at the top of the sidebar once the app is running. A short clip
with a decent amount of people crossing paths works best, since that's what
actually triggers occlusion / ID switch moments.

First run will also download the YOLOv8 weights for whatever model size you
pick (via `ultralytics`) - that can take a minute depending on your
connection, it's a one-time thing per model size after that.

## What each control does

(quick version - see `instructions.txt` for the full walkthrough)

- **model size dropdown** - picks which YOLOv8 checkpoint runs detection.
  bigger = catches more people in dense/packed areas, slower per frame.
- **confidence threshold slider** - filters detections live. higher = fewer
  but more reliable boxes, lower = more boxes but more junk.
- **frame skip slider** - how many frames to skip during extraction, for
  speed on longer clips.
- **run / restart playback button** - (re)runs detection + tracking with the
  current settings and plays the result back frame by frame in both panels.
- **process log (sidebar)** - type a quick note and it gets timestamped and
  appended to `process_log.txt`, alongside auto-logged notes whenever you
  change model/threshold. Meant to double as raw notes for the actual
  writeup.

## Known limitations

- playback isn't a real video player, it's `st.image` updated in a loop
  frame by frame, so it can look a little choppy on bigger clips - that's
  expected, not a bug.
- `yolov8x` (and even `yolov8l`) are genuinely slow on CPU, expect to wait.
- ByteTrack params (IoU threshold, max age, low-conf floor) are reasonable
  defaults, not tuned for any particular clip - if ID switches still happen
  a lot on your video, that's probably why.
- SORT and ByteTrack currently share one global track-ID counter under the
  hood, so the ID numbers in each panel won't both start at 1 - doesn't
  affect correctness, just looks a little odd.
- the "ID switch" / "fragmentation" metrics are a practical proxy (spatial
  overlap + gap heuristics), not the official MOTChallenge metric definitions
  - there's no ground truth for uploaded clips, so this is the best local
  approximation.
- whole video gets loaded into memory as frames up front, so very long clips
  will be slow to start and use more RAM than you'd want.
