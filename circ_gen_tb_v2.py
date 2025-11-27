import os, random, cv2
import numpy as np

SYMBOL_MAP = {
	"res": "A",
	"cap": "B",
	"ind": "C",
	"cgen": "D",
	"vgen": "E",
	"diode": "F",
}

def clear_folder(folder):
	os.makedirs(folder, exist_ok=True)
	for fname in os.listdir(folder):
		fpath = os.path.join(folder, fname)
		if os.path.isfile(fpath):
			os.remove(fpath)

def resize_to_height(img, target_height):
	h, w = img.shape[:2]
	if h == 0 or w == 0:
		raise ValueError("Üres kép érkezett a resize-hoz.")
	scale = target_height / h
	new_w = max(1, int(w * scale))
	return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)

def generate_rotations(img):
	if img is None or img.size == 0:
		raise ValueError("Üres kép a rotációhoz.")
	rots = []
	rots.append(img)  # 0°
	rots.append(cv2.flip(img, 1))  # 0° + H flip
	r90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
	rots.append(r90)
	rots.append(cv2.flip(r90, 1))
	r180 = cv2.rotate(img, cv2.ROTATE_180)
	rots.append(r180)
	rots.append(cv2.flip(r180, 1))
	r270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
	rots.append(r270)
	rots.append(cv2.flip(r270, 1))
	return rots

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
	return img

def load_symbols_with_rotations(image_folder):
	items = []
	for fname in os.listdir(image_folder):
		if not fname.lower().endswith(".jpg"):
			continue
		path = os.path.join(image_folder, fname)
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		if img is None or img.size == 0:
			continue
		base = os.path.splitext(fname)[0].lower()
		for key, label in SYMBOL_MAP.items():
			if key in base:
				for r in generate_rotations(img):
					items.append((r, label))
				break
	if not items:
		raise ValueError("Nincs betöltött szimbólum (ellenőrizd a fájlneveket és a mappát).")
	return items

def generate_toolbar_files(image_folder, tiff_folder, jpg_folder,
						   num_files=50, line_length=10,
						   jpg_quality=95, use_char_boxes=False,
						   augment=False, small_rotation=False,
						   random_background=False):
	symbols = load_symbols_with_rotations(image_folder)

	# Gyors elérés label -> lista
	by_label = {}
	for img, lbl in symbols:
		by_label.setdefault(lbl, []).append(img)

	for idx in range(num_files):
		if idx < len(SYMBOL_MAP):
			# első len(SYMBOL_MAP) sor: véletlen permutáció, minden label egyszer
			perm_labels = np.random.permutation(list(SYMBOL_MAP.values()))
			line_labels = list(perm_labels)
		else:
			# normál random: line_length darab
			all_labels = list(SYMBOL_MAP.values())
			line_labels = [random.choice(all_labels) for _ in range(line_length)]

		# képek kiválasztása a címkékhez
		line_imgs = []
		for lbl in line_labels:
			candidates = by_label.get(lbl, [])
			if not candidates:
				raise ValueError(f"Nincs példa a(z) '{lbl}' labelhez.")
			line_imgs.append((random.choice(candidates), lbl))

		# egységes magasság
		target_height = max(img.shape[0] for img, _ in line_imgs)
		line_imgs_resized = [(resize_to_height(img, target_height), lbl) for img, lbl in line_imgs]

		# darabonkénti augmentáció
		if augment:
			line_imgs_resized = [(augment_image(img, small_rotation=small_rotation), lbl)
								 for img, lbl in line_imgs_resized]

		# összefűzés
		pieces = [img for img, _ in line_imgs_resized]
		if not pieces:
			raise ValueError("Üres darablista a fűzésnél.")
		aug_row = np.hstack(pieces)
		if aug_row is None or aug_row.size == 0:
			raise ValueError("Üres összefűzött kép keletkezett.")

		# JPG mentés (színes)
		jpg_path = os.path.join(jpg_folder, f"toolbar_{idx}.jpg")
		ok_jpg = cv2.imwrite(jpg_path, aug_row, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
		if not ok_jpg:
			raise IOError(f"JPG mentés sikertelen: {jpg_path}")

		# TIFF mentés (grayscale) + háttér variálás fehér pixelek átszínezésével
		gray = cv2.cvtColor(aug_row, cv2.COLOR_BGR2GRAY)
		if random_background:
			bg = np.random.randint(200, 256)
			# fehér (közel fehér) pixelek cseréje bg-re
			mask_white = gray > 250
			gray[mask_white] = bg

		tiff_path = os.path.join(tiff_folder, f"toolbar_{idx}.tif")
		ok_tif = cv2.imwrite(tiff_path, gray)
		if not ok_tif:
			raise IOError(f"TIFF mentés sikertelen: {tiff_path}")

		# .box és .gt.txt
		box_path = os.path.join(tiff_folder, f"toolbar_{idx}.box")
		gt_path = os.path.join(tiff_folder, f"toolbar_{idx}.gt.txt")
		with open(box_path, "w", encoding="utf-8") as box_file, \
			 open(gt_path, "w", encoding="utf-8") as gt_file:

			x_offset = 0
			labels_out = []
			for img, lbl in line_imgs_resized:
				h, w = img.shape[:2]
				labels_out.append(lbl)

				if use_char_boxes:
					# szimbólumonkénti bounding box
					box_file.write(f"{lbl} {x_offset} 0 {x_offset+w} {h} 0\n")

				x_offset += w

			gt_string = "".join(labels_out)
			gt_file.write(gt_string + "\n")

			if not use_char_boxes:
				# minden karakter külön sorban, de a teljes sor befoglaló méretével
				total_w = aug_row.shape[1]
				total_h = aug_row.shape[0]
				for lbl in labels_out:
					box_file.write(f"{lbl} 0 0 {total_w} {total_h} 0\n")

	print(f"[INFO] {num_files} sor generálva: {tiff_folder} (TIFF/BOX/GT) és {jpg_folder} (JPG).")

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
		use_char_boxes=True,
		augment=True,
		small_rotation=True,
		random_background=True
	)
