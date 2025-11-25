import os, random, cv2, unicodedata
from PIL import Image
import numpy as np

# --- Symbol mapping (Latin letters only) ---
# Each component name is mapped to a unique Latin letter.
SYMBOL_MAP = {
    "res":   "A",   # Resistance
    "cap":   "B",   # Capacitor
    "ind":   "C",   # Inductor
    "cgen":  "D",   # Current generator
    "vgen":  "E",   # Voltage generator
    "diode": "F",   # Diode
}

def clear_folder(folder):
    """
    Delete all files inside the given folder.
    The folder itself is preserved, created if missing.
    """
    if os.path.exists(folder):
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
    else:
        os.makedirs(folder, exist_ok=True)

def generate_toolbar_files(image_folder, tiff_folder, jpg_folder,
                           num_files=50, line_length=10,
                           jpg_quality=95, use_char_boxes=False):
    """
    Generate toolbar images and training files:
    - TIFF (grayscale) for training
    - GT text file with Latin letters
    - BOX file (either line-level or character-level depending on use_char_boxes)
    - JPG (RGB) preview for visual inspection

    Parameters:
    - use_char_boxes: if True, generate per-character bounding boxes
                      if False, generate line-level boxes (whole image for each char)
    """

    # Ensure output folders exist and are empty
    os.makedirs(tiff_folder, exist_ok=True)
    os.makedirs(jpg_folder, exist_ok=True)
    clear_folder(tiff_folder)
    clear_folder(jpg_folder)

    keys = list(SYMBOL_MAP.keys())

    for n in range(num_files):
        # Randomly select symbols for one toolbar line
        chosen = random.choices(keys, k=line_length)
        textline = "".join(SYMBOL_MAP[s] for s in chosen)

        # Load each symbol image from image_folder (expects <name>.jpg)
        imgs = [Image.open(os.path.join(image_folder, f"{s}.jpg")).convert("RGB") for s in chosen]
        widths, heights = zip(*(im.size for im in imgs))
        total_w, max_h = sum(widths), max(heights)

        # Create RGB toolbar image (for preview)
        toolbar_rgb = Image.new("RGB", (total_w, max_h), (255, 255, 255))
        x = 0
        for im in imgs:
            toolbar_rgb.paste(im, (x, 0))
            x += im.size[0]

        base = f"toolbar_{n}"

        # --- Save TIFF (grayscale) ---
        toolbar_gray = toolbar_rgb.convert("L")
        tif_path = os.path.join(tiff_folder, base + ".tif")
        gt_path  = os.path.join(tiff_folder, base + ".gt.txt")
        box_path = os.path.join(tiff_folder, base + ".box")

        toolbar_gray.save(tif_path, format="TIFF", compression="none", dpi=(300, 300))

        # --- Save GT text file ---
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(textline + "\n")

        # --- Save BOX file ---
        w, h = toolbar_gray.size
        box_lines = []

        if use_char_boxes:
            # Character-level bounding boxes
            x_offset = 0
            for ch, im in zip(textline, imgs):
                cw, ch_h = im.size
                # Coordinates: left, bottom, right, top
                # Note: Tesseract uses bottom-left origin
                box_lines.append(f"{ch} {x_offset} 0 {x_offset+cw} {ch_h} 0")
                x_offset += cw
        else:
            # Line-level boxes (whole image for each character)
            box_lines = [f"{ch} 0 0 {w} {h} 0" for ch in textline]

        with open(box_path, "w", encoding="utf-8") as f:
            f.write("\n".join(box_lines) + "\n")

        # --- Save JPG preview (RGB) ---
        jpg_path = os.path.join(jpg_folder, base + ".jpg")
        toolbar_rgb.save(jpg_path, format="JPEG", quality=jpg_quality)

        #print(f"Saved: {tif_path}, {jpg_path} | GT={textline} | BOX={box_path}")

jpg_path = "Symbols jpg"
jpg_toolbar_path = "Toolbars jpg"
tiff_path = "data/circuit-ground-truth"   

generate_toolbar_files(jpg_path, tiff_path, jpg_toolbar_path, num_files=2000, line_length=15, use_char_boxes=True)

