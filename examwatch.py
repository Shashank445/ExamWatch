import cv2
import math
from ultralytics import YOLO
import cvzone
import numpy as np
from collections import defaultdict
import os
import sys

# Load YOLO model (will download automatically if not present)
print("📥 Loading YOLO11m Pose model...")
model = YOLO('yolo11m-pose.pt')

# Configuration: Update these paths to point to your video file
# You can use a local video file or set to 0 for webcam
if len(sys.argv) > 1:
    source_path = sys.argv[1]
else:
    source_path = "classroom.mp4"  # Default video file, or use 0 for webcam

output_path = "output_cheating_pose_video.mp4"

# Check if video file exists, otherwise use webcam
if isinstance(source_path, str) and source_path != "0" and not os.path.exists(source_path):
    print(f"⚠️  Video file '{source_path}' not found. Switching to webcam (0)...")
    source_path = 0
elif source_path == "0":
    source_path = 0

# Open video source (file or webcam)
print(f"🎥 Opening video source: {source_path}")
cap = cv2.VideoCapture(source_path if isinstance(source_path, str) else int(source_path))

if not cap.isOpened():
    print(f"❌ Error: Could not open video source: {source_path}")
    print("💡 Tip: Make sure the video file exists, or use 0 for webcam")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

print(f"📐 Video dimensions: {width}x{height} @ {fps} FPS")

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_id = 0
print("🎥 Running improved detection...")

next_id = 0
tracker = {}
MAX_MATCH_DIST = 80
SIDEWAYS_PERSISTENCE = 5
YAW_DEG_THRESHOLD = 15
PITCH_PIX_THRESHOLD = 18

def compute_pose_metrics(kpts):
    try:
        nose = kpts[0]
        left_eye = kpts[1]
        right_eye = kpts[2]
        left_shoulder = kpts[5]
        right_shoulder = kpts[6]
        shoulder_cx = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0]) + 1e-6
        yaw_ratio = (nose[0] - shoulder_cx) / shoulder_width
        yaw_deg = math.degrees(math.atan(yaw_ratio))
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        pitch_pixels = nose[1] - eye_mid_y
        return float(yaw_deg), float(pitch_pixels)
    except Exception as e:
        return None, None

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def match_to_tracker(detections_centers):
    global tracker, next_id
    assigned = {}
    used_tracks = set()
    for di, c in enumerate(detections_centers):
        best_id, best_dist = None, 1e9
        for tid, state in tracker.items():
            if tid in used_tracks:
                continue
            tcenter = state['center']
            dist = math.hypot(c[0] - tcenter[0], c[1] - tcenter[1])
            if dist < best_dist:
                best_dist, best_id = dist, tid
        if best_id is not None and best_dist < MAX_MATCH_DIST:
            assigned[di] = best_id
            used_tracks.add(best_id)
        else:
            assigned[di] = None
    return assigned

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Processing complete.")
            break

        frame_id += 1
        results = model(frame, stream=True)
        detections = []

        for r in results:
            for box, keypoints in zip(r.boxes, r.keypoints):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if label == "person" and conf > 0.45:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    kpts = keypoints.xy[0].cpu().numpy()
                    yaw_deg, pitch_pixels = compute_pose_metrics(kpts)
                    center = get_center((x1, y1, x2, y2))
                    detections.append(((x1, y1, x2, y2), yaw_deg, pitch_pixels, center))

        centers = [d[3] for d in detections]
        mapping = match_to_tracker(centers) if len(centers) > 0 and len(tracker) > 0 else {i: None for i in range(len(centers))}

        seen_ids = set()
        for di, det in enumerate(detections):
            box, yaw_deg, pitch_pixels, center = det
            assigned_id = mapping.get(di, None)
            if assigned_id is None:
                assigned_id = next_id
                next_id += 1
                tracker[assigned_id] = {'box': box, 'center': center, 'sideways_count': 0, 'last_seen': frame_id}
            else:
                tracker[assigned_id]['box'] = box
                tracker[assigned_id]['center'] = center
                tracker[assigned_id]['last_seen'] = frame_id
            seen_ids.add(assigned_id)

            sideways_confirmed = False
            looking_down = False
            yaw_val = yaw_deg if yaw_deg is not None else 0.0
            pitch_val = pitch_pixels if pitch_pixels is not None else 0.0

            if pitch_val is not None and pitch_val > PITCH_PIX_THRESHOLD:
                looking_down = True
                tracker[assigned_id]['sideways_count'] = 0
            else:
                if yaw_deg is not None and abs(yaw_val) > YAW_DEG_THRESHOLD:
                    tracker[assigned_id]['sideways_count'] = tracker[assigned_id].get('sideways_count', 0) + 1
                else:
                    tracker[assigned_id]['sideways_count'] = 0
                if tracker[assigned_id]['sideways_count'] >= SIDEWAYS_PERSISTENCE:
                    sideways_confirmed = True

            x1, y1, x2, y2 = box
            if looking_down:
                color = (0, 200, 0)
                label_text = "Reading (down)"
            elif sideways_confirmed:
                color = (0, 0, 255)
                label_text = "Looking Sideways"
            else:
                color = (0, 255, 120)
                label_text = "Focused"

            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), colorC=color)
            cvzone.putTextRect(frame, f"ID:{assigned_id} {label_text}", (x1, max(0, y1 - 25)), scale=0.8, thickness=1, colorR=color)
            debug_txt = f"yaw:{yaw_val:.1f}" if yaw_deg is not None else "yaw:NA"
            debug_txt2 = f"p:{pitch_val:.1f}" if pitch_pixels is not None else "p:NA"
            cv2.putText(frame, debug_txt + " " + debug_txt2, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        to_delete = []
        for tid, st in tracker.items():
            if st['last_seen'] < frame_id - 40:
                to_delete.append(tid)
        for tid in to_delete:
            del tracker[tid]

        out.write(frame)

        if frame_id % 40 == 0:
            print(f"🟢 Processed {frame_id} frames...")
        
        # Display frame (press 'q' to quit)
        cv2.imshow('ExamWatch - Cheating Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️  Stopped by user")
            break

except KeyboardInterrupt:
    print("\n⏹️  Interrupted by user")
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("🎉 Output video saved successfully to:", output_path)

