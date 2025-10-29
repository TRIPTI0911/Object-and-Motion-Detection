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

# Global flag for stopping the script
RUNNING = True

def stop_motion_detector():
    global RUNNING
    RUNNING = False

def run_motion_detector():
    global RUNNING
    RUNNING = True

    # --- Your original code starts here (unchanged except inside function) ---

    model = YOLO("yolov8n.pt")

    base_dir = "motion_detector"
    snapshot_dir = os.path.join(base_dir, "Snapshots")
    video_dir = os.path.join(base_dir, "Videos")
    log_file = os.path.join(base_dir, "motion_log.csv")
    body_log_file = os.path.join(base_dir, "body_movement_log.csv")

    for d in [base_dir, snapshot_dir, video_dir]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(log_file):
        pd.DataFrame(columns=["Timestamp", "Object"], dtype=str).to_csv(log_file, index=False)
    if not os.path.exists(body_log_file):
        pd.DataFrame(columns=["Timestamp", "Movement"], dtype=str).to_csv(body_log_file, index=False)

    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

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
    snapshot_interval = 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None
    video_path = None

    fgbg = cv2.createBackgroundSubtractorMOG2()
    tracker = Tracker(distance_function="euclidean", distance_threshold=30)
    movement_last = {}

    def detect_movement(part, current_pos, threshold=2):
        prev = movement_last.get(part)
        movement_last[part] = current_pos
        if prev is None:
            return False
        return np.linalg.norm(np.array(prev) - np.array(current_pos)) > threshold

    movement_events = []
    object_log_entries = []

    def log_body_movement(label):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        movement_events.append({"Timestamp": timestamp, "Movement": label})

    def log_object_motion(label):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        object_log_entries.append({"Timestamp": timestamp, "Object": label})

    now = time.time()
    for folder in [snapshot_dir, video_dir]:
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path) and os.stat(path).st_mtime < now - 86400:
                os.remove(path)

    frame_count = 0

    while cap.isOpened() and RUNNING:
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = mp.solutions.face_mesh.FaceMesh().process(rgb)
        hand_result = mp.solutions.hands.Hands().process(rgb)
        pose_result = mp.solutions.pose.Pose().process(rgb)

        h, w, _ = frame.shape
        if face_result.multi_face_landmarks:
            face_landmarks = face_result.multi_face_landmarks[0].landmark
            lip_coords = [(face_landmarks[13].x * w, face_landmarks[13].y * h),
                          (face_landmarks[14].x * w, face_landmarks[14].y * h)]
            lips_center = np.mean(lip_coords, axis=0)
            if detect_movement("lips", lips_center):
                cv2.putText(frame, "Lips Moving", (10, 40), font, 0.7, (255, 0, 0), 2)
                log_body_movement("Lips Moving")

        cv2.imshow("Motion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
