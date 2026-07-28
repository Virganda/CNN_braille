import cv2
import numpy as np
import os
from processor import BrailleProcessor

def debug_save_crops(image_path, output_dir='debug_crops'):
    os.makedirs(output_dir, exist_ok=True)
    p = BrailleProcessor()
    
    img, thresh = p.preprocess(image_path)
    dots = p.detect_dots(thresh)
    cell_groups = p.group_to_cells(dots)
    
    count = 0
    for row_idx, row in enumerate(cell_groups):
        for cell_idx, cell_dots in enumerate(row):
            crop = p.get_cell_crop(img, cell_dots)
            # Remove channel dim and save
            crop_img = (crop.squeeze() * 255).astype('uint8')
            filename = f"row{row_idx}_cell{cell_idx}.png"
            cv2.imwrite(os.path.join(output_dir, filename), crop_img)
            count += 1
            
    print(f"Saved {count} crops to {output_dir}")

if __name__ == "__main__":
    # Use any recently uploaded image if available, else skip
    upload_dir = 'uploads'
    if os.path.exists(upload_dir):
        files = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(('.jpg', '.png'))]
        if files:
            latest = max(files, key=os.path.getmtime)
            print(f"Analyzing latest upload: {latest}")
            debug_save_crops(latest)
        else:
            print("No uploads found.")
    else:
        print("Uploads directory missing.")
