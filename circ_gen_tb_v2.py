import os, random, cv2, unicodedata
import numpy as np

# --- Symbol mapping (Latin letters only) ---
SYMBOL_MAP = {
    "res":   "A",   # Resistance
    "cap":   "B",   # Capacitor
    "ind":   "C",   # Inductor
    "cgen":  "D",   # Current generator
    "vgen":  "E",   # Voltage generator
    "diode": "F",   # Diode
}

def clear_folder(folder):
    if os.path.exists(folder):
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
    else:
        os.makedirs(folder, exist_ok=True)

def resize_to_height(img, target_height):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

def generate_rotations(img):
    rotations = []
    rotations.append(img)
    rotations.append(cv2.flip(img, 1))
    rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotations.append(rot90)
    rotations.append(cv2.flip(rot90, 1))
    rot180 = cv2.rotate(img, cv2.ROTATE_180)
    rotations.append(rot180)
    rotations.append(cv2.flip(rot180, 1))
    rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotations.append(rot270)
    rotations.append(cv2.flip(rot270, 1))
    return rotations

def augment_image(img, small_rotation=False):
    choice = np.random.randint(0, 6)
    if choice == 0:
        return img
    elif choice == 1:
        return cv2.GaussianBlur(img, (3, 3), 0)
    elif choice == 2:
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif choice == 3:
        return cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    elif choice == 4:
        return cv2.convertScaleAbs(img, alpha=1.3, beta=0)
    elif choice == 5 and small_rotation:
        angle = np.random.uniform(-10, 10)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    else:
        return img

def generate_toolbar_files(image_folder, tiff_folder, jpg_folder,
                           num_files=50, line_length=10,
                           jpg_quality=95, use_char_boxes=False,
                           augment=False, small_rotation=False,
                           random_background=False):
    symbols = []
    for fname in os.listdir(image_folder):
        if fname.lower().endswith(".jpg"):
            img = cv2.imread(os.path.join(image_folder, fname), cv2.IMREAD_COLOR)
            base = os.path.splitext(fname)[0].lower()
            for key, val in SYMBOL_MAP.items():
                if key in base:
                    for rot in generate_rotations(img):
                        symbols.append((rot, val))
                    break

    for idx in range(num_files):
        if idx < len(SYMBOL_MAP):
            perm = np.random.permutation(list(SYMBOL_MAP.values()))
            line_imgs = []
            for label in perm:
                candidates = [img for img, lbl in symbols if lbl == label]
                line_imgs.append((random.choice(candidates), label))
        else:
            line_imgs = [symbols[np.random.randint(len(symbols))] for _ in range(line_length)]

        target_height = max(img.shape[0] for img, _ in line_imgs)
        line_imgs_resized = [(resize_to_height(img, target_height), label) for img, label in line_imgs]

        # augmentáció darabonként
        if augment:
            line_imgs_resized = [(augment_image(img, small_rotation=small_rotation), label)
                                 for img, label in line_imgs_resized]

        aug = np.hstack([img for img, _ in line_imgs_resized])

        # háttér variálás sor szinten
        bg_color = np.random.randint(200, 256) if random_background else 255
        canvas = np.full((aug.shape[0], aug.shape[1], 3), bg_color, dtype=np.uint8)
        mask = aug > 0
        canvas[mask] = aug[mask]
        aug = canvas

        jpg_path = os.path.join(jpg_folder, f"toolbar_{idx}.jpg")
        tiff_path = os.path.join(tiff_folder, f"toolbar_{idx}.tif")
        cv2.imwrite(jpg_path, aug, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        gray_aug = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(tiff_path, gray_aug)

        box_path = os.path.join(tiff_folder, f"toolbar_{idx}.box")
        gt_path = os.path.join(tiff_folder, f"toolbar_{idx}.gt.txt")
        with open(box_path, "w", encoding="utf-8") as box_file, \
             open(gt_path, "w", encoding="utf-8") as gt_file:
            x_offset = 0
            labels = []
            for img, char_label in line_imgs_resized:
                h, w = img.shape[:2]
                labels.append(char_label)
                if use_char_boxes:
                    box_file.write(f"{char_label} {x_offset} 0 {x_offset+w} {h} 0\n")
                x_offset += w
            gt_file.write("".join(labels) + "\n")

    print(f"[INFO] {num_files} toolbar sor generálva a {tiff_folder} és {jpg_folder} mappákba.")

# Példahívás
if __name__ == "__main__":
    jpg_path = "Symbols jpg"
    jpg_toolbar_path = "Toolbars jpg"
    tiff_path = "data/circuit-ground-truth"
    clear_folder(jpg_toolbar_path)
    clear_folder(tiff_path)
    generate_toolbar_files(
        image_folder=jpg_path,
        tiff_folder=tiff_path,
        jpg_folder=jpg_toolbar_path,
        num_files=1000,
        line_length=15,
        jpg_quality=95,
        use_char_boxes=False,
        augment=True,
        small_rotation=True,
        random_background=True
    )
