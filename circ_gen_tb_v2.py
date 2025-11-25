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

def resize_to_height(img, target_height):
    """
    Átméretezi a képet úgy, hogy a magasság target_height legyen,
    az arányok megtartásával.
    """
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

def augment_image(img):
    """
    Véletlenszerű augmentációk alkalmazása egy képre.
    (Elforgatás nélkül)
    """
    choice = np.random.randint(0, 5)  # 0–4
    if choice == 0:
        return img
    elif choice == 1:
        # 🌫️ Blur
        return cv2.GaussianBlur(img, (3, 3), 0)
    elif choice == 2:
        # 🔊 Zaj hozzáadása
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif choice == 3:
        # ☀️ Fényerő növelése
        return cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    elif choice == 4:
        # 🎚️ Kontraszt növelése
        return cv2.convertScaleAbs(img, alpha=1.3, beta=0)

def generate_toolbar_files(image_folder, tiff_folder, jpg_folder,
                           num_files=50, line_length=10,
                           jpg_quality=95, use_char_boxes=False,
                           augment=False):
    """
    Toolbar képek generálása TIFF (grayscale) + JPG (színes) formátumban,
    opcionális augmentációval.
    """
    symbols = []
    for fname in os.listdir(image_folder):
        if fname.lower().endswith(".jpg"):
            img = cv2.imread(os.path.join(image_folder, fname), cv2.IMREAD_COLOR)  # színes beolvasás
            base = os.path.splitext(fname)[0].lower()
            for key, val in SYMBOL_MAP.items():
                if key in base:   # rugalmasabb ellenőrzés
                    symbols.append((img, val))
                    break

    for idx in range(num_files):
        # véletlen sor összeállítása (img, label párokkal)
        line_imgs = [symbols[np.random.randint(len(symbols))] for _ in range(line_length)]

        # célmagasság: a sor legmagasabb szimbóluma
        target_height = max(img.shape[0] for img, _ in line_imgs)

        # minden szimbólumot átméretezünk erre a magasságra
        line_imgs_resized = [(resize_to_height(img, target_height), label) for img, label in line_imgs]

        # összefűzés
        aug = np.hstack([img for img, _ in line_imgs_resized])

        # augmentáció opcionálisan
        if augment:
            aug = augment_image(aug)

        # fájlnevek
        jpg_path = os.path.join(jpg_folder, f"toolbar_{idx}.jpg")
        tiff_path = os.path.join(tiff_folder, f"toolbar_{idx}.tif")

        # JPG mentés színesen
        cv2.imwrite(jpg_path, aug, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

        # TIFF mentés grayscale-ben
        gray_aug = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(tiff_path, gray_aug)

        # .box és .gt.txt fájlok generálása
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
                    # karakterhatárok megadása
                    box_file.write(f"{char_label} {x_offset} 0 {x_offset+w} {h} 0\n")

                x_offset += w

            # ground truth sor
            gt_file.write("".join(labels) + "\n")

    # függvény vége
    print(f"[INFO] {num_files} toolbar sor generálva a {tiff_folder} és {jpg_folder} mappákba.")

# Példahívás
if __name__ == "__main__":
    jpg_path = "Symbols jpg"
    jpg_toolbar_path = "Toolbars jpg"
    tiff_path = "data/circuit-ground-truth"

    # előkészítés: mappák ürítése
    clear_folder(jpg_toolbar_path)
    clear_folder(tiff_path)

    generate_toolbar_files(
        image_folder=jpg_path,        # bemeneti szimbólumok mappa
        tiff_folder=tiff_path,        # TIFF fájlok célmappa (grayscale)
        jpg_folder=jpg_toolbar_path,  # JPG fájlok célmappa (színes)
        num_files=1000,               # sorok száma
        line_length=15,               # soronkénti szimbólumok száma
        jpg_quality=95,               # JPG minőség
        use_char_boxes=True,          # karakter szintű box fájlok
        augment=True                  # augmentáció bekapcsolva
    )
