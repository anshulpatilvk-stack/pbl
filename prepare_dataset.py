import os
import shutil
import csv
import cv2
import numpy as np

RAW_DIR = "raw_datasets"
TRAIN_FRESH = os.path.join("dataset", "train", "fresh")
TRAIN_ROTTEN = os.path.join("dataset", "train", "rotten")
CSV_REPORT = "dataset_freshness_scores.csv"

os.makedirs(TRAIN_FRESH, exist_ok=True)
os.makedirs(TRAIN_ROTTEN, exist_ok=True)

ROTTEN_KEYWORDS = ["rotten", "stale", "bad", "overripe", "spoiled", "diseased", "defect", "mold", "rot"]
FRESH_KEYWORDS = ["fresh", "good", "healthy", "ripe", "unripe"]
TARGET_FRUITS = ["apple", "banana", "lemon", "tomato", "my_dataset", "anshul", "content", "dataset"]


def score_image_0_to_100(img_path: str) -> int:
    """Computes a 0-100 freshness score using OpenCV HSV vibrancy and rot spot detection."""
    img = cv2.imread(img_path)
    if img is None:
        return -1

    img_resized = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Isolate fruit pixels from plain backgrounds
    fg_mask = cv2.inRange(hsv, np.array([0, 35, 20]), np.array([180, 255, 245]))
    total_pixels = np.count_nonzero(fg_mask)
    if total_pixels < 500:
        fg_mask = np.ones((224, 224), dtype=np.uint8) * 255
        total_pixels = 224 * 224

    # Vibrant fresh colors (Red, Orange, Yellow, Green)
    fresh_r1 = cv2.inRange(hsv, np.array([0, 65, 70]), np.array([12, 255, 255]))
    fresh_r2 = cv2.inRange(hsv, np.array([160, 65, 70]), np.array([180, 255, 255]))
    fresh_yg = cv2.inRange(hsv, np.array([14, 60, 75]), np.array([85, 255, 255]))
    fresh_mask = cv2.bitwise_and(cv2.bitwise_or(cv2.bitwise_or(fresh_r1, fresh_r2), fresh_yg), fg_mask)

    # Dark rot and brown blemish spots
    dark_rot = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))
    brown_rot = cv2.inRange(hsv, np.array([8, 30, 35]), np.array([25, 165, 135]))
    spoilage_mask = cv2.bitwise_and(cv2.bitwise_or(dark_rot, brown_rot), fg_mask)

    fresh_ratio = np.count_nonzero(fresh_mask) / total_pixels
    spoilage_ratio = np.count_nonzero(spoilage_mask) / total_pixels
    mean_sat = np.mean(s[fg_mask > 0]) / 255.0

    raw_score = (0.65 * fresh_ratio) + (0.35 * mean_sat) - (1.45 * spoilage_ratio) + 0.25
    score_100 = int(round(np.clip(raw_score * 100, 0, 100)))
    return score_100


fresh_count = 0
rotten_count = 0

with open(CSV_REPORT, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Filename", "Original_Path", "Classification", "Freshness_Score_0_100"])

    for root, _, files in os.walk(RAW_DIR):
        # Skip YOLO folder if it does not contain classification crops
        if "yolo" in root.lower():
            continue

        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            full_path = os.path.join(root, file)
            path_lower = full_path.lower()

            # Ensure we only process relevant fruit folders
            if not any(k in path_lower for k in TARGET_FRUITS):
                continue

            cv_score = score_image_0_to_100(full_path)
            if cv_score == -1:
                continue

            # Determine Good (fresh) vs Bad (rotten) from folder name first, fallback to CV score
            if any(k in path_lower for k in ROTTEN_KEYWORDS):
                label = "Bad (Rotten)"
                final_score = min(cv_score, 40)  # Keep known rotten items in 0-40 range
                dest_folder = TRAIN_ROTTEN
                rotten_count += 1
            elif any(k in path_lower for k in FRESH_KEYWORDS):
                label = "Good (Fresh)"
                final_score = max(cv_score, 65)  # Keep known fresh items in 65-100 range
                dest_folder = TRAIN_FRESH
                fresh_count += 1
            else:
                # Unlabeled personal photos: classify automatically via 0-100 CV score
                if cv_score >= 50:
                    label = "Good (Fresh)"
                    final_score = cv_score
                    dest_folder = TRAIN_FRESH
                    fresh_count += 1
                else:
                    label = "Bad (Rotten)"
                    final_score = cv_score
                    dest_folder = TRAIN_ROTTEN
                    rotten_count += 1

            unique_name = f"{fresh_count + rotten_count}_{file}"
            shutil.copy2(full_path, os.path.join(dest_folder, unique_name))
            writer.writerow([unique_name, full_path, label, final_score])

print(f"Done! Sorted {fresh_count} Good (Fresh) images and {rotten_count} Bad (Rotten) images.")
print(f"Full 0-100 score log saved to {CSV_REPORT}")