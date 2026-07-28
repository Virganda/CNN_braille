import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class BrailleProcessor:
    def __init__(self, cell_size=(28, 28)):
        self.cell_size        = cell_size
        self.median_cell_dim  = 0
        self.avg_cell_width   = 0       
        self._intercell_gaps  = []      
        self._word_gap_thresh = None    
        self.row_bounds       = []      

    # ------------------------------------------------------------------ #
    #  PREPROCESSING (Anti-Fusi: Tanpa Dilasi Berlebih)
    # ------------------------------------------------------------------ #
    def preprocess(self, image_path, show_plot=False):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur dikecilkan agar titik tidak memudar ke sekitarnya
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold dipertajam untuk mengisolasi titik dengan presisi tinggi
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 45, 12
        )

        # KUNCI PERBAIKAN: Hapus Dilasi (Penebalan). 
        # Kita HANYA membersihkan noise debu kertas dengan Morph Opening ringan.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.subplot(1, 2, 2)
            plt.title("Final Threshold (Cek apakah titik terpisah rapi)")
            plt.imshow(final_thresh, cmap='gray')
            plt.tight_layout()
            plt.show()

        print("Image loaded   :", image_path)
        print("Original shape :", img.shape)
        
        return img, final_thresh

    # ------------------------------------------------------------------ #
    #  DOT DETECTION
    # ------------------------------------------------------------------ #
    def detect_dots(self, thresh):
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        raw_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            # Filter area diubah sedikit untuk toleransi titik kecil
            if area < 10 or area > 1000:
                continue
                
            aspect_ratio = float(w) / h if h > 0 else 0
            # Filter bentuk diperketat agar tidak membaca garis lecek kertas
            if 0.4 < aspect_ratio < 2.5:
                raw_candidates.append({
                    'rect'  : (x, y, w, h),
                    'center': (x + w // 2, y + h // 2),
                    'area'  : area,
                })

        if not raw_candidates:
            return []

        areas = sorted([c['area'] for c in raw_candidates])
        median_area = np.median(areas)
        min_area = max(10, median_area * 0.2)

        dots = []
        for c in raw_candidates:
            if c['area'] >= min_area:
                dots.append({
                    'rect'  : c['rect'],
                    'center': c['center'],
                })

        print(f"[detect_dots] {len(dots)} dots lolos filter akhir")
        return dots

    # ------------------------------------------------------------------ #
    #  CELL GROUPING (Dengan Safety Cap pada Dimensi)
    # ------------------------------------------------------------------ #
    def group_to_cells(self, dots):
        if not dots: return []

        widths  = [d['rect'][2] for d in dots]
        heights = [d['rect'][3] for d in dots]
        raw_avg_dim = (np.median(widths) + np.median(heights)) / 2

        # KUNCI PERBAIKAN: Safety Cap. 
        # Jika ada blob nyasar yg besar, avg_dim dikunci maksimal 22 piksel.
        avg_dim = min(max(raw_avg_dim, 8.0), 22.0)

        # ── STEP 1: SUB-BARIS ──
        dots_by_y = sorted(dots, key=lambda d: d['center'][1])
        dot_rows = []
        for d in dots_by_y:
            added = False
            for row in dot_rows:
                row_y = np.mean([x['center'][1] for x in row])
                if abs(d['center'][1] - row_y) < avg_dim * 1.0: 
                    row.append(d)
                    added = True
                    break
            if not added:
                dot_rows.append([d])

        dot_rows = sorted(dot_rows, key=lambda r: np.mean([x['center'][1] for x in r]))

        # ── STEP 2: Y-SPLIT (Pemisah Baris Absolut) ──
        if len(dot_rows) <= 1:
            text_lines = [dot_rows[0]] if dot_rows else []
        else:
            row_ys = [np.mean([d['center'][1] for d in r]) for r in dot_rows]
            y_split_thresh = avg_dim * 2.5 

            text_lines = []
            current_text_line = []
            current_text_line.extend(dot_rows[0])
            
            for i in range(1, len(dot_rows)):
                gap = row_ys[i] - row_ys[i-1]
                if gap < y_split_thresh:
                    current_text_line.extend(dot_rows[i])
                else:
                    text_lines.append(current_text_line)
                    current_text_line = []
                    current_text_line.extend(dot_rows[i])
            text_lines.append(current_text_line)

        text_lines = [line for line in text_lines if len(line) >= 2]

        self.row_bounds = []
        for line_dots in text_lines:
            rys = [d['center'][1] for d in line_dots]
            self.row_bounds.append({
                'min': min(rys),
                'max': max(rys),
                'range': max(rys) - min(rys)
            })

        # ── STEP 3: X-SPLIT (Segmentasi Huruf Absolut) ──
        structured_cells = []
        x_split_thresh = avg_dim * 2.2 

        for line_dots in text_lines:
            dots_by_x = sorted(line_dots, key=lambda d: d['center'][0])
            row_cells = []
            current_cell = [dots_by_x[0]]

            for i in range(1, len(dots_by_x)):
                gap_x = dots_by_x[i]['center'][0] - dots_by_x[i-1]['center'][0]
                if gap_x < x_split_thresh:
                    current_cell.append(dots_by_x[i])
                else:
                    row_cells.append(current_cell)
                    current_cell = [dots_by_x[i]]
            row_cells.append(current_cell)
            structured_cells.append(row_cells)

        # Hitung ukuran sel standar untuk spasi kata
        all_cell_dims = []
        for row in structured_cells:
            for cell in row:
                xs = [d['center'][0] for d in cell]
                all_cell_dims.append(max(xs) - min(xs))
        
        self.median_cell_dim = avg_dim * 3.0
        self.avg_cell_width = np.median([w for w in all_cell_dims if w > 0]) if all_cell_dims else avg_dim * 1.5
        
        # Mencegah perhitungan spasi negatif
        self.avg_cell_width = min(max(self.avg_cell_width, avg_dim), avg_dim * 2.5)

        # ── STEP 4: SPASI KATA ──
        self._word_gap_thresh = avg_dim * 4.0

        print(f"[group_to_cells] Tepat memisahkan {len(text_lines)} baris teks vertikal!")
        return structured_cells

    # ------------------------------------------------------------------ #
    #  CELL CROP FOR CNN
    # ------------------------------------------------------------------ #
    def get_cell_crop(self, img, cell_dots):
        if not cell_dots: return np.zeros((28, 28, 1), dtype='float32')

        center_x = int(np.mean([d['center'][0] for d in cell_dots]))
        center_y = int(np.mean([d['center'][1] for d in cell_dots]))
        side     = max(10, int(self.median_cell_dim * 1.6))
        nx, ny   = center_x - side // 2, center_y - side // 2

        h, w     = img.shape[:2]
        nx1, ny1 = max(0, nx),        max(0, ny)
        nx2, ny2 = min(w, nx + side), min(h, ny + side)

        crop = img[ny1:ny2, nx1:nx2]

        if crop.shape[0] < side or crop.shape[1] < side:
            pad_h = max(0, side - crop.shape[0])
            pad_w = max(0, side - crop.shape[1])
            crop  = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        crop = cv2.resize(crop, (28, 28)).astype('float32') / 255.0
        return np.expand_dims(crop, axis=-1)

    # ------------------------------------------------------------------ #
    #  RULE-BASED DOT PATTERN
    # ------------------------------------------------------------------ #
    def get_dot_pattern(self, thresh, cell_dots):
        if not cell_dots:
            return '000000'

        xs       = [d['center'][0] for d in cell_dots]
        ys       = [d['center'][1] for d in cell_dots]
        cy       = np.mean(ys)
        
        widths  = [d['rect'][2] for d in cell_dots]
        heights = [d['rect'][3] for d in cell_dots]
        raw_avg_dim = (np.median(widths) + np.median(heights)) / 2
        avg_dim = min(max(raw_avg_dim, 8.0), 22.0)

        x_spread = max(xs) - min(xs)
        
        if x_spread < avg_dim * 1.2:
            col_split = max(xs) + 1 
        else:
            col_split = (min(xs) + max(xs)) / 2.0

        matched_bound = None
        for b in self.row_bounds:
            if b['min'] - (avg_dim * 2) <= cy <= b['max'] + (avg_dim * 2):
                matched_bound = b
                break

        if matched_bound:
            row_center_y = matched_bound['min'] + (matched_bound['range'] / 2.0)
        else:
            row_center_y = cy

        y_top = row_center_y - (avg_dim * 0.8)
        y_bot = row_center_y + (avg_dim * 0.8)

        dots_found = [0] * 6
        for dot in cell_dots:
            x, y = dot['center']
            
            col  = 0 if x <= col_split else 1
            if y <= y_top: row = 0
            elif y > y_bot: row = 2
            else: row = 1
                
            dots_found[row + col * 3] = 1

        return "".join(map(str, dots_found))

if __name__ == "__main__":
    test_dir = "data/test"
    if os.path.exists(test_dir):
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image = os.path.join(test_dir, image_files[0])
            print(f"Mencoba memproses: {test_image}")
            p = BrailleProcessor()
            p.preprocess(test_image, show_plot=True)
        else:
            print("Tidak ada gambar.")
    else:
        print("Folder tidak ditemukan.")