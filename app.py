import os
import re # <-- Tambahan untuk NLP Text Normalization
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

from processor import BrailleProcessor
from translator import BrailleTranslator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

processor    = BrailleProcessor()
translator   = BrailleTranslator()
model        = None
model_labels = []

# ── Label loading ─────────────────────────────────────────────────────────
def load_labels():
    global model_labels
    labels_path = 'model_labels.txt'
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            model_labels = [l.strip() for l in f if l.strip()]
        print(f"Labels loaded: {len(model_labels)} classes")
    else:
        print("model_labels.txt not found — AI mode disabled")

# ── Model loading ─────────────────────────────────────────────────────────
def load_cnn():
    global model
    model_path = 'braille_cnn.h5'
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            print(f"CNN model loaded from {model_path}  "
                f"(output={model.output_shape[-1]}, labels={len(model_labels)})")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print("Model file not found — rule-based fallback will be used")

load_labels()
load_cnn()


# ── Label -> character mapping ────────────────────────────────────────────
_SPECIAL = {
    'Unlabeled'    : '',
    'period'       : '.',
    'question mark': '?',
    'capital'      : '^',
    'comma'        : ',',
    'exclamation'  : '!',
    'number'       : '#',
    'apostrophe'   : "'",
    'colon'        : ':',
    'semicolon'    : ';',
    'hyphen'       : '-',
}

def map_label_to_char(label):
    if label in _SPECIAL:
        return _SPECIAL[label]
    if len(label) == 1 and label.isalpha():
        return label.lower()
    return label


# ── FIX: Label CNN yang sering salah prediksi ─────────────────────────────
# Menambahkan 'question mark' dan 'o' agar otomatis ditangani oleh rule-based
_CNN_UNRELIABLE = {'v', 'd', 'x', 'question mark', 'o', 'period', 'y', 'e'}


# ── Hybrid prediction: CNN + rule-based fallback ──────────────────────────
def predict_cell(orig_img, thresh, cell_dots):
    if not cell_dots:
        return '', 'empty'

    xs       = [d['center'][0] for d in cell_dots]
    x_spread = max(xs) - min(xs)
    fallback = processor.median_cell_dim if processor.median_cell_dim > 0 else 30

    # Single-column cell -> langsung rule-based (CNN tidak reliable untuk ini)
    is_single_col = x_spread < fallback * 0.25
    if is_single_col:
        pattern = processor.get_dot_pattern(thresh, cell_dots)
        char    = translator.translate(pattern)
        return char, f'rule(single-col) pattern={pattern}'

    # Multi-column cell -> CNN dulu
    if model is not None and model_labels:
        cell_crop = processor.get_cell_crop(orig_img, cell_dots)
        pred      = model.predict(np.expand_dims(cell_crop, axis=0), verbose=0)[0]
        idx       = int(np.argmax(pred))
        conf      = float(pred[idx])

        # Syarat confidence dinaikkan menjadi 0.85
        if conf >= 0.85 and idx < len(model_labels):
            label = model_labels[idx]

            # Skip label yang CNN-nya tidak reliable
            if label != 'Unlabeled' and label not in _CNN_UNRELIABLE:
                char = map_label_to_char(label)
                return char, f'CNN label={label} conf={conf:.2f}'
            elif label in _CNN_UNRELIABLE:
                # CNN prediksi karakter yang tidak reliable -> paksa rule-based
                pattern = processor.get_dot_pattern(thresh, cell_dots)
                char    = translator.translate(pattern)
                return char, f'rule(cnn-unreliable label={label} conf={conf:.2f}) pattern={pattern}'

        # CNN tidak confident -> rule-based
        pattern = processor.get_dot_pattern(thresh, cell_dots)
        char    = translator.translate(pattern)
        return char, f'rule(low-conf={conf:.2f}) pattern={pattern}'

    # Tidak ada model -> rule-based
    pattern = processor.get_dot_pattern(thresh, cell_dots)
    char    = translator.translate(pattern)
    return char, f'rule(no-model) pattern={pattern}'


# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        orig_img, thresh = processor.preprocess(filepath)
        dots             = processor.detect_dots(thresh)
        structured_cells = processor.group_to_cells(dots)

        full_text = ""
        debug_img = orig_img.copy()

        for row_idx, row in enumerate(structured_cells):
            row_text   = ""
            is_capital = False

            for cell_idx, cell_dots in enumerate(row):
                # Gambar debug bounding box
                if cell_dots:
                    min_x = min(d['rect'][0] for d in cell_dots)
                    max_x = max(d['rect'][0] + d['rect'][2] for d in cell_dots)
                    min_y = min(d['rect'][1] for d in cell_dots)
                    max_y = max(d['rect'][1] + d['rect'][3] for d in cell_dots)
                    cv2.rectangle(debug_img,
                                (min_x - 2, min_y - 2), (max_x + 2, max_y + 2),
                                (0, 255, 0), 1)

                char, method = predict_cell(orig_img, thresh, cell_dots)
                print(f"  [CELL] row={row_idx} cell={cell_idx} {method} -> '{char}'")

                # Terapkan aturan Braille
                if char == '^':
                    is_capital = True
                elif char == ' ':
                    row_text  += ' '
                    is_capital = False
                elif char:
                    if is_capital:
                        row_text  += char.upper()
                        is_capital = False
                    else:
                        row_text += char

                # Evaluasi GAP untuk spasi kata dengan estimated boundary
                if cell_idx < len(row) - 1 and cell_dots and row[cell_idx + 1]:
                    half_w = processor.avg_cell_width / 2.0
                    if half_w < 1:
                        half_w = processor.median_cell_dim * 0.4  # safety

                    curr_cx    = np.median([d['center'][0] for d in cell_dots])
                    next_cx    = np.median([d['center'][0] for d in row[cell_idx + 1]])
                    curr_right = curr_cx + half_w   # estimated right boundary
                    next_left  = next_cx - half_w   # estimated left boundary
                    gap        = next_left - curr_right

                    word_thresh = (
                        processor._word_gap_thresh
                        if processor._word_gap_thresh is not None
                        else processor.median_cell_dim * 2.5   # fallback safety
                    )

                    print(f"  [GAP]  cell {cell_idx}→{cell_idx+1}: "
                        f"norm_gap={gap:.1f}px  thresh={word_thresh:.1f}px  "
                        f"{'→ SPASI KATA' if gap > word_thresh else ''}")

                    if gap > word_thresh:
                        row_text += ' '

            if row_text.strip():
                full_text += row_text + "\n"

        debug_path = os.path.join('static', 'debug_processed.jpg')
        cv2.imwrite(debug_path, debug_img)

        # ── POST-PROCESSING: TEXT NORMALIZATION (ORGANIK) ────────────
        # 1. Hapus simbol noise optik (seperti kutip tunggal nyasar).
        # Hanya menyisakan huruf, angka, spasi, dan tanda baca dasar (.,?!)
        clean_text = re.sub(r'[^\w\s.,?!-]', '', full_text)
        
        # 2. Rapikan spasi ganda di dalam baris (tanpa merusak baris baru/enter)
        clean_lines = []
        for line in clean_text.split('\n'):
            line = re.sub(r'[ \t]+', ' ', line).strip()
            if line:
                clean_lines.append(line)
        
        final_clean_text = '\n'.join(clean_lines)
        # ─────────────────────────────────────────────────────────────

        return jsonify({
            'text'       : final_clean_text,
            'char_count' : len(final_clean_text),
            'line_count' : len(final_clean_text.split('\n')) if final_clean_text else 0,
            'debug_image': '/' + debug_path
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        from gtts import gTTS
        import io
        from flask import send_file

        tts = gTTS(text=text, lang='id', slow=False)
        fp  = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return send_file(fp, mimetype='audio/mp3')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)