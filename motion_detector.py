import cv2
import os
import time
import shutil
import datetime
import numpy as np
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace
from norfair import Detection, Tracker, draw_tracked_objects

# Set up YOLO model
model = YOLO("yolov8n.pt")

# Create output folders
base_dir = "motion_detector"
snapshot_dir = os.path.join(base_dir, "Snapshots")
video_dir = os.path.join(base_dir, "Videos")
log_file = os.path.join(base_dir, "motion_log.csv")
body_log_file = os.path.join(base_dir, "body_movement_log.csv")

for d in [base_dir, snapshot_dir, video_dir]:
    os.makedirs(d, exist_ok=True)

# Initialize CSV logs
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Timestamp", "Object"], dtype=str).to_csv(log_file, index=False)
if not os.path.exists(body_log_file):
    pd.DataFrame(columns=["Timestamp", "Movement"], dtype=str).to_csv(body_log_file, index=False)

# Initialize video capture
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose()
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()

font = cv2.FONT_HERSHEY_SIMPLEX
motion_detected = False
recording = False
motion_start_time = None
last_snapshot_time = 0
snapshot_interval = 2  # seconds
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None
video_path = None

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# DeepSORT Tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Movement tracking dictionary
movement_last = {}
def detect_movement(part, current_pos, threshold=2):
    prev = movement_last.get(part)
    movement_last[part] = current_pos
    if prev is None:
        return False
    return np.linalg.norm(np.array(prev) - np.array(current_pos)) > threshold

# Log body movement to CSV
movement_events = []
def log_body_movement(label):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    movement_events.append({"Timestamp": timestamp, "Movement": label})

# Log object motion to CSV
object_log_entries = []
def log_object_motion(label):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    object_log_entries.append({"Timestamp": timestamp, "Object": label})

# Cleanup snapshots older than 1 day
now = time.time()
for folder in [snapshot_dir, video_dir]:
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and os.stat(path).st_mtime < now - 86400:
            os.remove(path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    mask = fgbg.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = any(cv2.contourArea(c) > 1000 for c in contours)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if motion:
        if not recording:
            motion_start_time = datetime.datetime.now()
            video_path = os.path.join(video_dir, f"video_{timestamp}.mp4")
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True

        if time.time() - last_snapshot_time > snapshot_interval:
            snapshot_path = os.path.join(snapshot_dir, f"snapshot_{timestamp}.jpg")
            cv2.imwrite(snapshot_path, frame)
            last_snapshot_time = time.time()

        if video_writer:
            video_writer.write(frame)

    else:
        if recording:
            if video_writer:
                video_writer.release()
                video_writer = None
            recording = False

    # YOLO object detection + DeepSORT tracking
    results = model.predict(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = tuple(np.random.randint(100, 255, size=3).tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        detections.append(Detection(points=np.array([cx, cy]), scores=np.array([1.0]), label=label))

        if motion and label not in ["person"]:
            log_object_motion(label)

    tracked_objects = tracker.update(detections=detections)

    for obj in tracked_objects:
        x, y = obj.estimate[0]
        obj_id = obj.id
        label = obj.label if hasattr(obj, "label") else "object"
        text = f"{label.title()} {obj_id}"

        # Emotion detection (only every 10 frames to avoid lag)
        if label == "person":
            try:
                for box in results.boxes:
                    if results.names[int(box.cls[0])] == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.shape[0] > 0 and face_img.shape[1] > 0 and frame_count % 10 == 0:
                            analysis = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False, detector_backend='opencv')
                            emotion = analysis[0]['dominant_emotion']
                            text = f"Person {obj_id} ({emotion})"
                        break
            except Exception:
                pass

        cv2.putText(frame, text, (int(x), int(y) - 10), font, 0.7, (0, 0, 255), 2)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    draw_tracked_objects(frame, tracked_objects)

    # Mediapipe face/hands/pose landmarks + movement label detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)
    pose_result = pose.process(rgb)

    h, w, _ = frame.shape
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0].landmark

        # Lips movement
        lip_coords = [(face_landmarks[13].x * w, face_landmarks[13].y * h),
                      (face_landmarks[14].x * w, face_landmarks[14].y * h)]
        lips_center = np.mean(lip_coords, axis=0)
        if detect_movement("lips", lips_center):
            cv2.putText(frame, "Lips Moving", (10, 40), font, 0.7, (255, 0, 0), 2)
            log_body_movement("Lips Moving")

        # Eyes movement
        eye_coords = [(face_landmarks[159].x * w, face_landmarks[159].y * h),
                      (face_landmarks[145].x * w, face_landmarks[145].y * h)]
        eye_center = np.mean(eye_coords, axis=0)
        if detect_movement("eyes", eye_center):
            cv2.putText(frame, "Eyes Moving", (10, 70), font, 0.7, (0, 255, 255), 2)
            log_body_movement("Eyes Moving")

        # Head movement
        head_center = (face_landmarks[10].x * w, face_landmarks[10].y * h)
        if detect_movement("head", head_center):
            cv2.putText(frame, "Head Moving", (10, 100), font, 0.7, (255, 255, 0), 2)
            log_body_movement("Head Moving")

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            hand_center = (wrist.x * w, wrist.y * h)
            if detect_movement("hand", hand_center):
                cv2.putText(frame, "Hand Moving", (10, 130), font, 0.7, (0, 255, 0), 2)
                log_body_movement("Hand Moving")

    if pose_result.pose_landmarks:
        hair_landmark = pose_result.pose_landmarks.landmark[0]
        hair_center = (hair_landmark.x * w, hair_landmark.y * h)
        if detect_movement("hair", hair_center):
            cv2.putText(frame, "Hair Movement", (10, 160), font, 0.7, (255, 0, 255), 2)
            log_body_movement("Hair Movement")

    cv2.imshow("Motion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save movement logs
if movement_events:
    df = pd.read_csv(body_log_file)
    df = pd.concat([df, pd.DataFrame(movement_events)], ignore_index=True)
    df.to_csv(body_log_file, index=False)

if object_log_entries:
    df = pd.read_csv(log_file)
    df = pd.concat([df, pd.DataFrame(object_log_entries)], ignore_index=True)
    df.to_csv(log_file, index=False)

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
