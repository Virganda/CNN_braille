"""
Diagnosis tool: runs the full pipeline on an image and prints
exactly what each cell looks like, what pattern is detected,
and what the CNN predicts — so we can see where it goes wrong.

Usage:
    python3 scripts/diagnose.py [image_path]
    (defaults to the most recent upload)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import tensorflow as tf

from processor import BrailleProcessor
from translator import BrailleTranslator

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = 'braille_cnn.h5'
LABELS_PATH = 'model_labels.txt'
OUT_DIR     = 'debug_crops'
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_upload():
    d = 'uploads'
    files = [os.path.join(d, f) for f in os.listdir(d)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        raise FileNotFoundError("No uploads found")
    return max(files, key=os.path.getmtime)

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_upload()
    print(f"\n=== Diagnosing: {image_path} ===\n")

    # Load model & labels
    model, model_labels = None, []
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABELS_PATH) as f:
                model_labels = [l.strip() for l in f.read().splitlines()]
            print(f"CNN loaded — {len(model_labels)} classes: {model_labels}\n")
        except Exception as e:
            print(f"Could not load CNN: {e}\n")
    else:
        print("CNN not found — using rule-based fallback\n")

    p = BrailleProcessor()
    t = BrailleTranslator()

    img, thresh = p.preprocess(image_path)
    dots = p.detect_dots(thresh)
    print(f"Total dots detected: {len(dots)}")

    cells = p.group_to_cells(dots)
    print(f"Rows: {len(cells)}")
    for ri, row in enumerate(cells):
        print(f"  Row {ri}: {len(row)} cells")
    print(f"Median cell dim: {p.median_cell_dim:.1f}px\n")

    # Save annotated image
    os.makedirs(OUT_DIR, exist_ok=True)
    debug_img = img.copy()

    full_text = ""
    for ri, row in enumerate(cells):
        row_text = ""
        is_capital = False
        print(f"── Row {ri} ──────────────────────────────")
        for ci, cell_dots in enumerate(row):
            if not cell_dots:
                continue

            # Bounding box
            min_x = min(d['rect'][0] for d in cell_dots)
            max_x = max(d['rect'][0] + d['rect'][2] for d in cell_dots)
            min_y = min(d['rect'][1] for d in cell_dots)
            max_y = max(d['rect'][1] + d['rect'][3] for d in cell_dots)
            cv2.rectangle(debug_img, (min_x-2, min_y-2), (max_x+2, max_y+2), (0,255,0), 1)

            # Rule-based pattern
            pattern = p.get_dot_pattern(thresh, cell_dots)
            rule_char = t.translate(pattern)

            # CNN prediction
            ai_label, ai_conf, ai_char = "N/A", 0.0, ""
            if model and model_labels:
                crop = p.get_cell_crop(img, cell_dots)
                pred = model.predict(np.expand_dims(crop, axis=0), verbose=0)[0]
                idx = np.argmax(pred)
                ai_conf = pred[idx]
                ai_label = model_labels[idx]
                # Map label to char (same logic as app.py)
                special = {'period':'.','question mark':'?','capital':'^',
                           'comma':',','exclamation':'!','Unlabeled':''}
                ai_char = special.get(ai_label, ai_label.lower() if len(ai_label)==1 else ai_label)

            # Save crop
            crop_img = p.get_cell_crop(img, cell_dots)
            crop_vis = (crop_img.squeeze() * 255).astype('uint8')
            cv2.imwrite(f"{OUT_DIR}/r{ri}_c{ci}.png", crop_vis)

            # Determine final char (AI if available, else rule)
            final_char = ai_char if (model and ai_conf > 0.3) else rule_char

            print(f"  Cell {ci:2d} | dots={len(cell_dots)} | pattern={pattern} | "
                  f"rule='{rule_char}' | AI='{ai_label}'({ai_conf:.2f}) | "
                  f"FINAL='{final_char}'")

            # Apply capital indicator
            if final_char == '^':
                is_capital = True
            elif final_char:
                ch = final_char.upper() if is_capital else final_char
                row_text += ch
                is_capital = False

            # Spacing
            if ci < len(row) - 1 and cell_dots and row[ci+1]:
                curr_right = max(d['center'][0] for d in cell_dots)
                next_left  = min(d['center'][0] for d in row[ci+1])
                gap = next_left - curr_right
                if gap > p.median_cell_dim * 1.3:
                    row_text += " "

        print(f"  → Row text: '{row_text}'")
        if row_text.strip():
            full_text += row_text + "\n"

    print(f"\n=== FINAL TRANSLATION ===\n{full_text.strip()}\n")

    # Save annotated debug image
    out_path = os.path.join('static', 'debug_diagnosis.jpg')
    os.makedirs('static', exist_ok=True)
    cv2.imwrite(out_path, debug_img)
    print(f"Annotated image saved → {out_path}")
    print(f"Cell crops saved      → {OUT_DIR}/")

if __name__ == '__main__':
    main()
