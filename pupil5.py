import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

CALIBRATION_FRAMES = 60
SMOOTHING_WINDOW   = 8
THRESHOLD_MARGIN   = 0.06

RANGE_MULTIPLIER   = 1.7
CONFIRM_FRAMES     = 3
out_of_range_count = 0

calibrated         = False
calibration_data   = []
baseline_left      = 0.5
baseline_right     = 0.5
baseline_head_yaw  = 0.0   # head yaw at calibration center
range_left         = 0.06
range_right        = 0.06

left_ratio_buf     = deque(maxlen=SMOOTHING_WINDOW)
right_ratio_buf    = deque(maxlen=SMOOTHING_WINDOW)
head_yaw_buf       = deque(maxlen=SMOOTHING_WINDOW)

mesh_spec    = mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=0)
contour_spec = mp_drawing.DrawingSpec(color=(255, 100, 0),  thickness=1, circle_radius=1)
iris_spec    = mp_drawing.DrawingSpec(color=(0, 200, 255),  thickness=1, circle_radius=1)

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def put_text_shadowed(frame, text, pos, scale, color, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x+2, y+2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def get_ratio(pupil_center, inner_corner, outer_corner):
    total_width = np.linalg.norm(outer_corner - inner_corner)
    if total_width == 0:
        return 0.5
    return np.linalg.norm(pupil_center - inner_corner) / total_width

def get_head_yaw(mesh_points, img_w):
    """
    Estimates horizontal head turn using the nose tip and face-width landmarks.
    Nose tip  : 1
    Left edge : 234  (left cheek)
    Right edge: 454  (right cheek)

    Returns a value roughly in [-1, +1]:
      0   = facing straight
      -ve = head turned right
      +ve = head turned left
    """
    nose      = mesh_points[1].astype(float)
    left_edge = mesh_points[234].astype(float)
    right_edge= mesh_points[454].astype(float)

    face_width = np.linalg.norm(right_edge - left_edge)
    if face_width < 1e-6:
        return 0.0

    # How far nose is from the midpoint of the two cheek landmarks
    mid_x = (left_edge[0] + right_edge[0]) / 2.0
    yaw   = (nose[0] - mid_x) / face_width   # normalised -0.5 … +0.5
    return float(yaw)

def get_compensated_diff(raw_diff, head_yaw, baseline_yaw,
                         YAW_COMPENSATION=0.8):
    """
    Subtracts head-turn contribution from the raw gaze diff.
    YAW_COMPENSATION: how strongly to cancel head movement (0=none, 1=full).
    Tune between 0.6 – 1.0.
    """
    head_delta = head_yaw - baseline_yaw        # how much head moved from center
    compensated = raw_diff - head_delta * YAW_COMPENSATION
    return compensated

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
    BAR_W = 300
    BAR_H = 18
    MAX_D = 0.35
    bx    = (img_w - BAR_W) // 2
    by    = img_h - 45

    cv2.rectangle(frame, (bx, by), (bx + BAR_W, by + BAR_H), (50, 50, 50), -1)

    safe_l = int(bx + BAR_W/2 - (range_l / MAX_D) * (BAR_W/2))
    safe_r = int(bx + BAR_W/2 + (range_r / MAX_D) * (BAR_W/2))
    safe_l = max(bx, safe_l)
    safe_r = min(bx + BAR_W, safe_r)
    cv2.rectangle(frame, (safe_l, by), (safe_r, by + BAR_H), (0, 180, 0), -1)
    cv2.rectangle(frame, (bx, by), (bx + BAR_W, by + BAR_H), (180, 180, 180), 1)

    cx = bx + BAR_W // 2
    cv2.line(frame, (cx, by), (cx, by + BAR_H), (255, 255, 255), 1)

    needle_x     = int(cx + np.clip(avg_diff / MAX_D, -1, 1) * (BAR_W // 2))
    needle_color = (0, 0, 255) if is_out_of_range(avg_diff, range_l, range_r) \
                   else (255, 255, 255)
    cv2.line(frame, (needle_x, by - 3), (needle_x, by + BAR_H + 3), needle_color, 3)

    put_text_shadowed(frame, "L", (bx - 18,        by + 14), 0.5, (100, 200, 255))
    put_text_shadowed(frame, "R", (bx + BAR_W + 4, by + 14), 0.5, (100, 200, 255))
    put_text_shadowed(frame, "EYE RANGE (head-compensated)",
                      (bx - 20, by - 6), 0.42, (200, 200, 200), 1)

def draw_head_indicator(frame, head_yaw, baseline_yaw, img_w, img_h):
    """Small bar showing head turn separately so user can see compensation."""
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

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame      = cv2.flip(frame, 1)
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results    = face_mesh.process(rgb_frame)
    img_h, img_w = frame.shape[:2]

    # ── CALIBRATION ───────────────────────────────────────────────────────────
    if not calibrated:
        progress = len(calibration_data)
        pct      = int((progress / CALIBRATION_FRAMES) * 100)
        bar_w    = int((img_w - 60) * pct / 100)

        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                draw_face_mesh(frame, fl)

        put_text_shadowed(frame, "CALIBRATION: Look straight at screen",
                          (30, 50), 0.8, (0, 255, 255))
        bar_overlay = frame.copy()
        cv2.rectangle(bar_overlay, (30, 62), (img_w - 30, 88), (30, 30, 30), -1)
        cv2.rectangle(bar_overlay, (30, 62), (30 + bar_w,  88), (0, 255, 0),  -1)
        cv2.addWeighted(bar_overlay, 0.6, frame, 0.4, 0, frame)
        put_text_shadowed(frame, f"{pct}%",
                          (img_w // 2 - 20, 83), 0.65, (255, 255, 255))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
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
                    l_vals  = [d[0] for d in calibration_data]
                    r_vals  = [d[1] for d in calibration_data]
                    y_vals  = [d[2] for d in calibration_data]

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

        cv2.imshow('Proctoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ── TRACKING ──────────────────────────────────────────────────────────────
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

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
            raw_yaw = get_head_yaw(mesh_points, img_w)
            head_yaw_buf.append(raw_yaw)
            smooth_yaw = float(np.mean(head_yaw_buf))

            smooth_left  = float(np.mean(left_ratio_buf))
            smooth_right = float(np.mean(right_ratio_buf))

            # Raw diffs
            raw_left_diff  = smooth_left  - baseline_left
            raw_right_diff = smooth_right - baseline_right
            raw_avg_diff   = (raw_left_diff + raw_right_diff) / 2.0

            # ── HEAD-COMPENSATED DIFF ─────────────────────────────────
            # Subtract head movement so only true eye movement remains
            comp_diff = get_compensated_diff(raw_avg_diff, smooth_yaw,
                                             baseline_head_yaw)

            # ── RANGE DECISION on COMPENSATED diff ───────────────────
            if is_out_of_range(comp_diff, range_left, range_right):
                out_of_range_count += 1
            else:
                out_of_range_count  = 0

            alert = out_of_range_count >= CONFIRM_FRAMES

            if alert:
                final_gaze = "Looking RIGHT" if comp_diff < 0 else "Looking LEFT"
            else:
                final_gaze = "Center"

            left_label  = "Right" if raw_left_diff  < -range_left  else \
                          ("Left"  if raw_left_diff  >  range_right else "Center")
            right_label = "Right" if raw_right_diff < -range_left  else \
                          ("Left"  if raw_right_diff >  range_right else "Center")
            status_color = (0, 0, 255) if alert else (0, 255, 0)

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
                (30, 40), 0.65, (255, 255, 0))
            put_text_shadowed(frame,
                f"Right Eye : {right_label}  (diff {raw_right_diff:+.2f})",
                (30, 72), 0.65, (255, 255, 0))
            put_text_shadowed(frame,
                f"Gaze: {final_gaze}",
                (30, 120), 1.15, status_color, 3)
            if alert:
                put_text_shadowed(frame,
                    "!! CHEATING DETECTED !!",
                    (30, 165), 0.9, (0, 0, 255), 2)

            # Range bar uses compensated diff so it stays stable on head move
            draw_range_bar(frame, comp_diff, range_left, range_right, img_w, img_h)
            draw_head_indicator(frame, smooth_yaw, baseline_head_yaw, img_w, img_h)

            put_text_shadowed(frame,
                f"Range: +/-{range_left:.2f}  |  R = Recalibrate  |  Q = Quit",
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

    cv2.imshow('Proctoring', frame)

cap.release()
cv2.destroyAllWindows()