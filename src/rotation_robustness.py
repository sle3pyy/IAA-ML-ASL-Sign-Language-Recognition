"""
rotation_robustness.py

Live webcam script that measures SVM model failure rate per label
as the hand is rotated horizontally (0 to 90 degrees of roll).

Usage:
    python3 rotation_robustness.py --model asl_svm_model_v5.pkl --labels A B C

Controls:
    Press the KEY of the label you are currently showing (e.g. 'a', 'b', 'c').
    The active label is shown on screen. Hold the sign and rotate slowly.
    Press 'q' to quit and save the final report + chart.
    Press 'r' to reset all collected data.
"""

import cv2
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.feature_extraction import extract_features_from_landmarks

# ---- Constants ----------------------------------------------------------------

FEATURE_FIELDS = [
    'thumb_index_dist', 'index_middle_dist', 'middle_ring_dist',
    'ring_pinky_dist', 'thumb_middle_dist', 'thumb_ring_dist', 'thumb_pinky_dist',
    'thumb_curl', 'index_curl', 'middle_curl', 'ring_curl', 'pinky_curl',
    'thumb_y', 'index_y', 'middle_y', 'ring_y', 'pinky_y',
    'hand_rotation', 'thumb_angle', 'palm_tilt',
    'spread_index_ring', 'spread_thumb_pinky', 'y_variance'
]

BUCKET_SIZE = 10          # degrees per bucket
MAX_ANGLE   = 110         # max horizontal rotation tracked
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# ---- Rotation helpers ---------------------------------------------------------

def compute_horizontal_roll_deg(world_landmarks):
    """
    Compute the horizontal roll of the palm (rotation around the
    camera's depth / Z axis) using 3D world landmarks.

    Uses:
        L0  - Wrist
        L5  - Index MCP (base of index finger)
        L17 - Pinky MCP (base of pinky finger)

    Returns the angle in [0, 180] degrees.
    0  = hand flat / horizontal (palm facing camera or away)
    90 = hand fully sideways (profile view)
    The angle increases continuously without resetting past 90 degrees
    because we use abs(atan2(y, x)) on the full signed angle.
    """
    p0  = np.array([world_landmarks[0].x,  world_landmarks[0].y,  world_landmarks[0].z])
    p5  = np.array([world_landmarks[5].x,  world_landmarks[5].y,  world_landmarks[5].z])
    p17 = np.array([world_landmarks[17].x, world_landmarks[17].y, world_landmarks[17].z])

    # Horizontal roll = angle of the L5->L17 lateral axis projected onto XY plane
    lateral = p17 - p5
    lateral_xy = np.array([lateral[0], lateral[1], 0.0])
    lat_norm = np.linalg.norm(lateral_xy)
    if lat_norm < 1e-6:
        return None

    # Use abs(atan2(y, x)) — takes the signed angle of the full vector and
    # returns its magnitude.  This correctly tracks [0, 180] without resetting:
    #   [+, 0]  -> 0deg   (horizontal)
    #   [+, +]  -> 45deg  (tilting)
    #   [0, +]  -> 90deg  (vertical / sideways)
    #   [-, +]  -> 135deg (past vertical, continuing same direction)
    # Symmetric: [+,-] [0,-] [-,-] give the same magnitudes respectively,
    # meaning left-roll and right-roll are both tracked 0->180.
    angle_rad = abs(math.atan2(lateral_xy[1], lateral_xy[0]))
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def angle_to_bucket(angle_deg):
    """Map a degree value to the lower bound of its 10-degree bucket, capped at MAX_ANGLE."""
    bucket = int(angle_deg // BUCKET_SIZE) * BUCKET_SIZE
    return min(bucket, MAX_ANGLE)


def make_buckets():
    """Return the list of bucket lower-bounds from 0 to MAX_ANGLE inclusive."""
    return list(range(0, MAX_ANGLE + BUCKET_SIZE, BUCKET_SIZE))


# ---- Stats helpers ------------------------------------------------------------

def failure_rate(total, correct):
    if total == 0:
        return None
    return (total - correct) / total * 100.0


def build_stats_table(stats, labels, buckets):
    """
    Returns a list of formatted lines for console overlay.
    stats[label][bucket] = [total, correct]
    """
    lines = []
    header = f"{'Label':<8}" + "".join(f"{b:>5}deg" for b in buckets)
    lines.append(header)
    lines.append("-" * len(header))
    for lbl in sorted(labels):
        row = f"{lbl:<8}"
        for b in buckets:
            t, c = stats[lbl].get(b, [0, 0])
            fr = failure_rate(t, c)
            if fr is None:
                row += f"{'---':>6}"
            else:
                row += f"{fr:>5.1f}%"
        lines.append(row)
    return lines


# ---- Visualization overlay ----------------------------------------------------

def draw_overlay(img, active_label, angle_deg, bucket, stats, labels, buckets,
                 pred_label=None):
    h, w = img.shape[:2]
    overlay = img.copy()
    alpha = 0.55

    # Semi-transparent sidebar
    cv2.rectangle(overlay, (0, 0), (340, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    y = 30
    dy = 18

    # Active label indicator
    cv2.putText(img, f"Active label: {active_label if active_label else '(none)'}",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
    y += dy + 4

    # Model prediction in the sidebar
    if pred_label is not None:
        is_correct = active_label and str(pred_label).upper() == active_label.upper()
        pred_color = (0, 220, 0) if is_correct else (0, 60, 220)
        cv2.putText(img, f"Prediction: {pred_label}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pred_color, 1)
    else:
        cv2.putText(img, "Prediction: ---",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
    y += dy + 4

    # Current rotation angle
    calibrated = getattr(draw_overlay, '_calibrated', False)
    if angle_deg is not None:
        color = (0, 200, 255) if calibrated else (0, 150, 200)
        cal_tag = "" if calibrated else "  [press ENTER to calibrate]"
        cv2.putText(img, f"Roll: {angle_deg:.1f} deg  (bucket {bucket}){cal_tag}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    else:
        cv2.putText(img, "Roll: ---", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    y += dy + 6

    # Per-label failure table header
    cv2.putText(img, "Label | Bucket | Fail%", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    y += dy

    if active_label:
        for b in buckets:
            t, c = stats[active_label].get(b, [0, 0])
            if t == 0:
                continue
            fr = failure_rate(t, c)
            color = (0, 255 - int(fr * 2.55), int(fr * 2.55))
            text = f"{active_label}  {b:>2}d: {fr:>5.1f}% ({t})"
            cv2.putText(img, text, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            y += dy
            if y > h - 20:
                break

    # Large prediction badge on the frame (top-right corner)
    if pred_label is not None:
        is_correct = active_label and str(pred_label).upper() == active_label.upper()
        badge_color = (0, 200, 0) if is_correct else (0, 0, 200)
        badge_text = str(pred_label)
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
        bx = w - tw - 20
        by = th + 20
        cv2.rectangle(img, (bx - 10, by - th - 10), (bx + tw + 10, by + 10),
                      badge_color, -1)
        cv2.putText(img, badge_text, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)

    # Controls hint
    cv2.putText(img, "Key=label  ENTER=calibrate  SPACE=pause  R=reset  Q=quit", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    # Paused banner
    if getattr(draw_overlay, '_paused', False):
        banner = "PAUSED"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
        bx = (w - bw) // 2
        by = h // 2
        cv2.rectangle(img, (bx - 16, by - bh - 10), (bx + bw + 16, by + 14),
                      (0, 0, 0), -1)
        cv2.putText(img, banner, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 80, 255), 3)


# ---- Report / Chart -----------------------------------------------------------

def save_report(stats, labels, buckets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n_labels  = len(labels)
    n_buckets = len(buckets)

    # -- CSV / text report
    rows = []
    for lbl in sorted(labels):
        for b in buckets:
            t, c = stats[lbl].get(b, [0, 0])
            fr = failure_rate(t, c)
            rows.append({
                'label': lbl,
                'bucket_deg': b,
                'total': t,
                'correct': c,
                'failure_pct': round(fr, 2) if fr is not None else None
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'rotation_robustness.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV report saved to {csv_path}")

    # -- Line chart: one line per label
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_labels, 1)))

    for i, lbl in enumerate(sorted(labels)):
        xs, ys = [], []
        for b in buckets:
            t, c = stats[lbl].get(b, [0, 0])
            fr = failure_rate(t, c)
            if fr is not None:
                xs.append(b)
                ys.append(fr)
        if xs:
            ax.plot(xs, ys, marker='o', label=lbl, color=colors[i], linewidth=2)

    ax.set_title("SVM Failure Rate vs Horizontal Hand Rotation", fontsize=14, fontweight='bold')
    ax.set_xlabel("Horizontal Roll Angle (degrees)", fontsize=11)
    ax.set_ylabel("Failure Rate (%)", fontsize=11)
    ax.set_xticks(buckets)
    ax.set_ylim(0, 105)
    ax.legend(title="Label", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, 'rotation_robustness.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to {chart_path}")

    # -- Heatmap: labels x buckets
    data_matrix = np.full((n_labels, n_buckets), np.nan)
    sorted_labels = sorted(labels)
    for r, lbl in enumerate(sorted_labels):
        for c_idx, b in enumerate(buckets):
            t, c = stats[lbl].get(b, [0, 0])
            fr = failure_rate(t, c)
            if fr is not None:
                data_matrix[r, c_idx] = fr

    fig2, ax2 = plt.subplots(figsize=(max(8, n_buckets), max(4, n_labels)))
    masked = np.ma.masked_invalid(data_matrix)
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color='lightgrey')
    im = ax2.imshow(masked, vmin=0, vmax=100, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax2, label='Failure Rate (%)')

    ax2.set_xticks(range(n_buckets))
    ax2.set_xticklabels([f"{b}d" for b in buckets])
    ax2.set_yticks(range(n_labels))
    ax2.set_yticklabels(sorted_labels)
    ax2.set_xlabel("Horizontal Roll Bucket")
    ax2.set_ylabel("Label")
    ax2.set_title("Failure Rate Heatmap: Label vs Rotation", fontweight='bold')

    for r in range(n_labels):
        for c_idx in range(n_buckets):
            val = data_matrix[r, c_idx]
            if not np.isnan(val):
                ax2.text(c_idx, r, f"{val:.0f}%",
                         ha='center', va='center',
                         fontsize=8, color='black')

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'rotation_robustness_heatmap.png')
    fig2.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")


# ---- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live rotation robustness analysis for ASL SVM model"
    )
    parser.add_argument(
        "--model", required=True,
        help="Name of the saved SVM model file in src/models/ (e.g. asl_svm_model_v5.pkl)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels to track (e.g. A B C). Defaults to model classes."
    )
    args = parser.parse_args()

    # -- Paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR   = os.path.join(CURRENT_DIR, "models")
    OUTPUT_DIR  = os.path.join(MODEL_DIR, "rotation_analysis")
    model_path  = os.path.join(MODEL_DIR, args.model)

    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Determine labels
    if args.labels:
        labels = [l.upper() for l in args.labels]
    elif hasattr(model, 'classes_'):
        labels = [str(c) for c in model.classes_]
    else:
        labels = []
    print(f"Tracking labels: {labels}")

    # Rotation buckets
    buckets = make_buckets()

    # Stats store: stats[label][bucket] = [total, correct]
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    # -- MediaPipe detector (access raw result for world landmarks)
    mp_task_path = os.path.join(CURRENT_DIR, "utils", "hand_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=mp_task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    active_label       = labels[0] if labels else None
    paused             = False
    calibration_angle  = None   # raw angle at user's 0-degree position
    print("\nControls:")
    print("  Press a letter key to set the active label you are showing")
    print("  ENTER = calibrate (set current hand position as 0 degrees)")
    print("  SPACE = pause / resume data collection")
    print("  R     = reset all data")
    print("  Q     = quit and save report\n")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            angle_deg  = None
            bucket     = None
            pred_class = None

            if not paused:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result   = detector.detect(mp_image)

                if result.hand_landmarks and result.hand_world_landmarks:
                    hand_lm  = result.hand_landmarks[0]
                    world_lm = result.hand_world_landmarks[0]
                    h, w     = frame.shape[:2]

                    # Pixel landmarks for feature extraction
                    pixel_landmarks = [
                        (int(lm.x * w), int(lm.y * h)) for lm in hand_lm
                    ]

                    # Draw skeleton
                    for conn in HAND_CONNECTIONS:
                        cv2.line(frame, pixel_landmarks[conn[0]],
                                 pixel_landmarks[conn[1]], (255, 80, 0), 2)
                    for pt in pixel_landmarks:
                        cv2.circle(frame, pt, 4, (0, 255, 0), -1)

                    # Compute raw angle and apply calibration offset
                    raw_angle = compute_horizontal_roll_deg(world_lm)
                    if raw_angle is not None:
                        if calibration_angle is not None:
                            # Angle relative to calibrated zero, clamped to [0, MAX_ANGLE]
                            angle_deg = max(0.0, min(float(MAX_ANGLE),
                                           abs(raw_angle - calibration_angle)))
                        else:
                            angle_deg = raw_angle

                    # Always extract features and predict (even without active label)
                    features_dict = extract_features_from_landmarks(
                        pixel_landmarks, frame.shape
                    )
                    if features_dict:
                        feature_vector = [features_dict[f] for f in FEATURE_FIELDS]
                        feature_df     = pd.DataFrame([feature_vector],
                                                      columns=FEATURE_FIELDS)
                        pred_class = model.predict(feature_df)[0]

                        # Only accumulate stats when calibrated, have a bucket and active label
                        if angle_deg is not None and active_label and calibration_angle is not None:
                            bucket = angle_to_bucket(angle_deg)
                            stats[active_label][bucket][0] += 1
                            if str(pred_class).upper() == active_label.upper():
                                stats[active_label][bucket][1] += 1

            # Expose state flags to draw_overlay via function attributes
            draw_overlay._paused     = paused
            draw_overlay._calibrated = calibration_angle is not None

            # Draw overlay
            draw_overlay(frame, active_label, angle_deg, bucket,
                         stats, labels if labels else list(stats.keys()),
                         buckets, pred_label=pred_class)

            cv2.imshow("Rotation Robustness Analysis  [SPACE=pause  Q=quit  R=reset]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
                state = "PAUSED" if paused else "RESUMED"
                print(f"Collection {state}.")
            elif key == 13 or key == 10:
                if angle_deg is not None:
                    # Store the raw angle as the calibration baseline
                    raw_angle = compute_horizontal_roll_deg(world_lm) if (
                        result.hand_landmarks and result.hand_world_landmarks
                    ) else None
                    if raw_angle is not None:
                        calibration_angle = raw_angle
                        print(f"Calibrated: 0 deg = {calibration_angle:.1f} deg (raw)")
                    else:
                        print("No hand detected for calibration.")
                else:
                    print("No hand detected for calibration.")
            elif key == ord('r'):
                stats.clear()
                print("Data reset.")
            elif key != 255:
                pressed = chr(key).upper()
                if pressed in [l.upper() for l in labels]:
                    active_label = pressed
                    print(f"Active label set to: {active_label}")
                elif labels:
                    print(f"Key '{pressed}' is not in tracked labels {labels}.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    # -- Print summary table
    all_labels  = sorted(set(list(labels) + list(stats.keys())))
    print("\n" + "=" * 60)
    print("ROTATION ROBUSTNESS SUMMARY")
    print("=" * 60)
    lines = build_stats_table(stats, all_labels, buckets)
    for l in lines:
        print(l)

    # -- Save report and charts
    if any(stats[lbl] for lbl in stats):
        save_report(stats, all_labels, buckets, OUTPUT_DIR)
    else:
        print("No data collected. Nothing saved.")


if __name__ == "__main__":
    main()
