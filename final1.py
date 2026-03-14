import cv2
import os
import time
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CANDIDATE_ID  = "USER_101"
EVIDENCE_PATH = "evidence"
os.makedirs(EVIDENCE_PATH, exist_ok=True)

VIOLATION_WEIGHTS = {
    "PHONE":     50,
    "GAZE_AWAY":  8,
}

# ── MODELS ────────────────────────────────────────────────────────────────────
obj_model = YOLO('yolo11n.pt')

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
face_mesh    = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)

mesh_spec    = mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=0)
contour_spec = mp_drawing.DrawingSpec(color=(255, 100, 0),  thickness=1, circle_radius=1)
iris_spec    = mp_drawing.DrawingSpec(color=(0, 200, 255),  thickness=1, circle_radius=1)

# ── GAZE / CALIBRATION STATE ──────────────────────────────────────────────────
CALIBRATION_FRAMES = 60
SMOOTHING_WINDOW   = 8
THRESHOLD_MARGIN   = 0.06   # minimum range floor

RANGE_MULTIPLIER   = 1.7    # std * multiplier = dynamic range
CONFIRM_FRAMES     = 3      # consecutive out-of-range frames to trigger alert
out_of_range_count = 0

calibrated         = False
calibration_data   = []
baseline_left      = 0.5
baseline_right     = 0.5
baseline_head_yaw  = 0.0
range_left         = 0.06
range_right        = 0.06

left_ratio_buf  = deque(maxlen=SMOOTHING_WINDOW)
right_ratio_buf = deque(maxlen=SMOOTHING_WINDOW)
head_yaw_buf    = deque(maxlen=SMOOTHING_WINDOW)

# ── SCORE / COOLDOWN STATE ────────────────────────────────────────────────────
cheat_score     = 0
last_phone_time = 0
last_gaze_time  = 0
PHONE_COOLDOWN  = 2.0
GAZE_COOLDOWN   = 1.0

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def put_text_shadowed(frame, text, pos, scale, color, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def save_evidence(frame, reason):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{CANDIDATE_ID}_{reason}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(EVIDENCE_PATH, filename), frame)
    print(f"[EVIDENCE] Saved: {filename}")

def get_ratio(pupil_center, inner_corner, outer_corner):
    total_width = np.linalg.norm(outer_corner - inner_corner)
    if total_width == 0:
        return 0.5
    return np.linalg.norm(pupil_center - inner_corner) / total_width

def get_head_yaw(mesh_points, img_w):
    """
    Estimates horizontal head turn using nose tip vs cheek midpoint.
    Nose tip   : landmark 1
    Left cheek : landmark 234
    Right cheek: landmark 454
    Returns normalised value: 0 = straight, -ve = turned right, +ve = turned left
    """
    nose       = mesh_points[1].astype(float)
    left_edge  = mesh_points[234].astype(float)
    right_edge = mesh_points[454].astype(float)
    face_width = np.linalg.norm(right_edge - left_edge)
    if face_width < 1e-6:
        return 0.0
    mid_x = (left_edge[0] + right_edge[0]) / 2.0
    return float((nose[0] - mid_x) / face_width)

def get_compensated_diff(raw_diff, head_yaw, baseline_yaw,
                         YAW_COMPENSATION=0.8):
    """
    Removes head-turn contribution from the raw gaze diff so only
    true eye movement remains.
    YAW_COMPENSATION: 0 = no cancellation, 1 = full cancellation (tune 0.6–1.0)
    """
    head_delta  = head_yaw - baseline_yaw
    return raw_diff - head_delta * YAW_COMPENSATION

def is_out_of_range(avg_diff, range_l, range_r):
    return avg_diff < -range_l or avg_diff > range_r

def draw_face_mesh(frame, face_landmarks):
    overlay = frame.copy()
    mp_drawing.draw_landmarks(overlay, face_landmarks,
        mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None, connection_drawing_spec=mesh_spec)
    mp_drawing.draw_landmarks(overlay, face_landmarks,
        mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None, connection_drawing_spec=contour_spec)
    mp_drawing.draw_landmarks(overlay, face_landmarks,
        mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None, connection_drawing_spec=iris_spec)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

def draw_range_bar(frame, avg_diff, range_l, range_r, img_w, img_h):
    """Horizontal gauge showing compensated gaze diff vs safe zone."""
    BAR_W = 300
    BAR_H = 18
    MAX_D = 0.35
    bx    = (img_w - BAR_W) // 2
    by    = img_h - 45

    # background
    cv2.rectangle(frame, (bx, by), (bx + BAR_W, by + BAR_H), (50, 50, 50), -1)

    # green safe zone
    safe_l = max(bx, int(bx + BAR_W/2 - (range_l / MAX_D) * (BAR_W/2)))
    safe_r = min(bx + BAR_W, int(bx + BAR_W/2 + (range_r / MAX_D) * (BAR_W/2)))
    cv2.rectangle(frame, (safe_l, by), (safe_r, by + BAR_H), (0, 180, 0), -1)

    # border + centre line
    cv2.rectangle(frame, (bx, by), (bx + BAR_W, by + BAR_H), (180, 180, 180), 1)
    cx = bx + BAR_W // 2
    cv2.line(frame, (cx, by), (cx, by + BAR_H), (255, 255, 255), 1)

    # needle
    needle_x     = int(cx + np.clip(avg_diff / MAX_D, -1, 1) * (BAR_W // 2))
    needle_color = (0, 0, 255) if is_out_of_range(avg_diff, range_l, range_r) \
                   else (255, 255, 255)
    cv2.line(frame, (needle_x, by - 3), (needle_x, by + BAR_H + 3), needle_color, 3)

    put_text_shadowed(frame, "L", (bx - 18,        by + 14), 0.5, (100, 200, 255))
    put_text_shadowed(frame, "R", (bx + BAR_W + 4, by + 14), 0.5, (100, 200, 255))
    put_text_shadowed(frame, "EYE RANGE (head-compensated)",
                      (bx - 20, by - 6), 0.42, (200, 200, 200), 1)

def draw_head_indicator(frame, head_yaw, baseline_yaw, img_w, img_h):
    """Small bar top-right showing live head-turn delta."""
    BAR_W = 160
    BAR_H = 10
    MAX_Y = 0.20
    bx    = img_w - BAR_W - 20
    by    = img_h - 80

    cv2.rectangle(frame, (bx, by), (bx + BAR_W, by + BAR_H), (30, 30, 30), -1)
    cx = bx + BAR_W // 2
    cv2.line(frame, (cx, by), (cx, by + BAR_H), (120, 120, 120), 1)

    delta    = head_yaw - baseline_yaw
    needle_x = int(cx + np.clip(delta / MAX_Y, -1, 1) * (BAR_W // 2))
    cv2.line(frame, (needle_x, by - 2), (needle_x, by + BAR_H + 2), (0, 200, 255), 2)
    cv2.rectangle(frame, (bx, by), (bx + BAR_W, by + BAR_H), (100, 100, 100), 1)
    put_text_shadowed(frame, f"HEAD YAW {delta:+.2f}",
                      (bx, by - 6), 0.38, (0, 200, 255), 1)

def draw_score_bar(frame, score, img_w, img_h):
    """Risk score bar bottom-right."""
    MAX_SCORE = 300
    bar_max_w = 200
    bar_h     = 18
    filled    = int(min(score / MAX_SCORE, 1.0) * bar_max_w)
    bx        = img_w - bar_max_w - 20
    by        = img_h - 35

    bar_color = (0, 255, 0) if score < 80 else \
                (0, 165, 255) if score < 180 else (0, 0, 255)

    cv2.rectangle(frame, (bx, by), (bx + bar_max_w, by + bar_h), (40, 40, 40), -1)
    if filled > 0:
        cv2.rectangle(frame, (bx, by), (bx + filled, by + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_max_w, by + bar_h), (200, 200, 200), 1)
    put_text_shadowed(frame, f"RISK SCORE: {int(score)}", (bx, by - 8), 0.5, bar_color)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame      = cv2.flip(frame, 1)
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = face_mesh.process(rgb_frame)
    img_h, img_w = frame.shape[:2]

    now                = time.time()
    current_violations = []

    # ── 1. PHONE DETECTION ────────────────────────────────────────────────────
    obj_results = obj_model(frame, stream=True, classes=[67], conf=0.5, verbose=False)
    for r in obj_results:
        if len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                put_text_shadowed(frame, f"PHONE {conf:.2f}",
                                  (x1, y1 - 10), 0.65, (0, 255, 0))

            if now - last_phone_time > PHONE_COOLDOWN:
                cheat_score     += VIOLATION_WEIGHTS["PHONE"]
                last_phone_time  = now
                current_violations.append("PHONE")
                save_evidence(frame, "PHONE_DETECTED")

    # ── 2. CALIBRATION ────────────────────────────────────────────────────────
    if not calibrated:
        progress = len(calibration_data)
        pct      = int((progress / CALIBRATION_FRAMES) * 100)
        bar_w    = int((img_w - 60) * pct / 100)

        if mp_results.multi_face_landmarks:
            for fl in mp_results.multi_face_landmarks:
                draw_face_mesh(frame, fl)

        put_text_shadowed(frame, "CALIBRATION: Look straight at screen",
                          (30, 50), 0.8, (0, 255, 255))
        bar_overlay = frame.copy()
        cv2.rectangle(bar_overlay, (30, 62), (img_w - 30, 88), (30, 30, 30), -1)
        cv2.rectangle(bar_overlay, (30, 62), (30 + bar_w,  88), (0, 255, 0),  -1)
        cv2.addWeighted(bar_overlay, 0.6, frame, 0.4, 0, frame)
        put_text_shadowed(frame, f"{pct}%",
                          (img_w // 2 - 20, 83), 0.65, (255, 255, 255))

        if mp_results.multi_face_landmarks:
            for face_landmarks in mp_results.multi_face_landmarks:
                mesh_points = np.array([
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in face_landmarks.landmark
                ])
                l_ratio = get_ratio(
                    np.array(cv2.minEnclosingCircle(mesh_points[468:473])[0]),
                    mesh_points[133], mesh_points[33])
                r_ratio = get_ratio(
                    np.array(cv2.minEnclosingCircle(mesh_points[473:478])[0]),
                    mesh_points[362], mesh_points[263])
                yaw = get_head_yaw(mesh_points, img_w)
                calibration_data.append((l_ratio, r_ratio, yaw))

                if len(calibration_data) >= CALIBRATION_FRAMES:
                    l_vals = [d[0] for d in calibration_data]
                    r_vals = [d[1] for d in calibration_data]
                    y_vals = [d[2] for d in calibration_data]

                    baseline_left     = float(np.mean(l_vals))
                    baseline_right    = float(np.mean(r_vals))
                    baseline_head_yaw = float(np.mean(y_vals))

                    avg_std = (np.std(l_vals) + np.std(r_vals)) / 2.0
                    dynamic = float(np.clip(avg_std * RANGE_MULTIPLIER,
                                            THRESHOLD_MARGIN, 0.15))
                    range_left  = dynamic
                    range_right = dynamic
                    calibrated  = True
                    print(f"[Calibrated] L={baseline_left:.3f} R={baseline_right:.3f} "
                          f"HeadYaw={baseline_head_yaw:.3f} Range=±{dynamic:.3f}")

        draw_score_bar(frame, cheat_score, img_w, img_h)
        cv2.imshow('Proctoring System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ── 3. GAZE TRACKING (with head compensation + range system) ──────────────
    gaze_alert = False
    final_gaze = "Center"

    if mp_results.multi_face_landmarks:
        for face_landmarks in mp_results.multi_face_landmarks:

            draw_face_mesh(frame, face_landmarks)

            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in face_landmarks.landmark
            ])

            # Left Eye
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[468:473])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            left_ratio_buf.append(get_ratio(center_left, mesh_points[133], mesh_points[33]))

            # Right Eye
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[473:478])
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            right_ratio_buf.append(get_ratio(center_right, mesh_points[362], mesh_points[263]))

            # Head yaw (smoothed)
            head_yaw_buf.append(get_head_yaw(mesh_points, img_w))
            smooth_yaw = float(np.mean(head_yaw_buf))

            smooth_left  = float(np.mean(left_ratio_buf))
            smooth_right = float(np.mean(right_ratio_buf))

            # Raw diffs from personal baseline
            raw_left_diff  = smooth_left  - baseline_left
            raw_right_diff = smooth_right - baseline_right
            raw_avg_diff   = (raw_left_diff + raw_right_diff) / 2.0

            # Head-compensated diff (removes head-turn false positives)
            comp_diff = get_compensated_diff(raw_avg_diff, smooth_yaw,
                                             baseline_head_yaw)

            # Range-based decision on compensated diff
            if is_out_of_range(comp_diff, range_left, range_right):
                out_of_range_count += 1
            else:
                out_of_range_count  = 0

            gaze_alert = out_of_range_count >= CONFIRM_FRAMES

            if gaze_alert:
                final_gaze = "Looking RIGHT" if comp_diff < 0 else "Looking LEFT"
            else:
                final_gaze = "Center"

            # Score with cooldown
            if gaze_alert and (now - last_gaze_time > GAZE_COOLDOWN):
                cheat_score    += VIOLATION_WEIGHTS["GAZE_AWAY"]
                last_gaze_time  = now
                current_violations.append("GAZE_AWAY")
                save_evidence(frame, "GAZE_AWAY")

            left_label  = "Right" if raw_left_diff  < -range_left  else \
                          ("Left"  if raw_left_diff  >  range_right else "Center")
            right_label = "Right" if raw_right_diff < -range_left  else \
                          ("Left"  if raw_right_diff >  range_right else "Center")
            status_color = (0, 0, 255) if gaze_alert else (0, 255, 0)

            # Pupil + corner dots
            cv2.circle(frame, center_left,      int(l_radius), (255, 0, 255), 2)
            cv2.circle(frame, mesh_points[133], 4, (0, 255, 255), -1)
            cv2.circle(frame, mesh_points[33],  4, (0, 255, 255), -1)
            cv2.circle(frame, center_right,     int(r_radius), (255, 0, 255), 2)
            cv2.circle(frame, mesh_points[362], 4, (0, 255, 255), -1)
            cv2.circle(frame, mesh_points[263], 4, (0, 255, 255), -1)

            # ── HUD ───────────────────────────────────────────────────
            put_text_shadowed(frame,
                f"Left Eye  : {left_label}  (diff {raw_left_diff:+.2f})",
                (30, 40), 0.6, (255, 255, 0))
            put_text_shadowed(frame,
                f"Right Eye : {right_label}  (diff {raw_right_diff:+.2f})",
                (30, 68), 0.6, (255, 255, 0))
            put_text_shadowed(frame,
                f"Gaze: {final_gaze}",
                (30, 112), 1.1, status_color, 3)

            if current_violations:
                put_text_shadowed(frame,
                    f"!! {' | '.join(current_violations)} !!",
                    (30, 155), 0.85, (0, 0, 255), 2)

            # Range bar (compensated) + head indicator
            draw_range_bar(frame, comp_diff, range_left, range_right, img_w, img_h)
            draw_head_indicator(frame, smooth_yaw, baseline_head_yaw, img_w, img_h)

    # ── SCORE BAR + FOOTER ────────────────────────────────────────────────────
    draw_score_bar(frame, cheat_score, img_w, img_h)
    put_text_shadowed(frame,
        f"Range:+/-{range_left:.2f}  |  R=Recalibrate  |  Q=Quit",
        (30, img_h - 15), 0.45, (200, 200, 200), 1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        calibrated         = False
        calibration_data   = []
        out_of_range_count = 0
        left_ratio_buf.clear()
        right_ratio_buf.clear()
        head_yaw_buf.clear()
        print("[Recalibrating...]")

    cv2.imshow('Proctoring System', frame)

cap.release()
cv2.destroyAllWindows()
print(f"\n[REPORT] Candidate: {CANDIDATE_ID} | Final Risk Score: {int(cheat_score)}")