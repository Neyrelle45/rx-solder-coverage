import os
import cv2
import math
import numpy as np
import pandas as pd
import joblib
import argparse
from glob import glob

# Optimisation OpenCV
try:
    cv2.setNumThreads(1)
except:
    pass

# =========================
# UTILS
# =========================
def load_gray(p, contrast_limit=2.0):
    # Si p est un chemin (string)
    if isinstance(p, (str, os.PathLike)):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    else:
        # Si p est un objet de type fichier (Streamlit)
        p.seek(0) # On remet au début au cas où
        file_bytes = np.frombuffer(p.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img is None: return None
    if contrast_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=contrast_limit, tileGridSize=(8,8))
        img = clahe.apply(img)
    return img

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def compose_similarity(scale, rotation_deg, tx, ty, cx, cy):
    angle = math.radians(rotation_deg)
    a = scale * math.cos(angle)
    b = scale * math.sin(angle)
    M = np.array([
        [a, b, (1-a)*cx - b*cy + tx],
        [-b, a, b*cx + (1-a)*cy + ty]
    ], dtype=np.float32)
    return M

# ==========================================
# CŒUR DU MOTEUR IA
# ==========================================
def compute_features(img: np.ndarray) -> np.ndarray:
    imgf = img.astype(np.float32)
    f_int = imgf / 255.0
    f_clahe = apply_clahe(img).astype(np.float32) / 255.0

    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 1, cv2.NORM_MINMAX)

    edge_dir = np.abs(gx) / (np.abs(gy) + 5.0)
    edge_dir = cv2.normalize(np.clip(edge_dir, 0, 10), None, 0, 1, cv2.NORM_MINMAX)

    mean3 = cv2.blur(imgf, (3, 3)) / 255.0
    mean21 = cv2.blur(imgf, (21, 21)) / 255.0
    
    v_kernel = np.ones((25, 1), np.float32) / 25.0
    v_line = cv2.filter2D(f_int, -1, v_kernel)

    dog = cv2.normalize(cv2.GaussianBlur(imgf,(3,3),0) - cv2.GaussianBlur(imgf,(15,15),0), None, 0, 1, cv2.NORM_MINMAX)

    g_v = cv2.normalize(cv2.filter2D(imgf, cv2.CV_32F, cv2.getGaborKernel((15,15), 4.0, math.pi/2, 10.0, 0.5, 0)), None, 0, 1, cv2.NORM_MINMAX)
    g_h = cv2.normalize(cv2.filter2D(imgf, cv2.CV_32F, cv2.getGaborKernel((15,15), 4.0, 0, 10.0, 0.5, 0)), None, 0, 1, cv2.NORM_MINMAX)

    return np.stack([f_int, f_clahe, mag, edge_dir, dog, v_line, mean3, mean21, g_v, g_h], axis=-1)

# =========================
# ENTRAÎNEMENT OPTIMISÉ
# =========================
def train_model(img_dir, lbl_dir, out_dir, n_estimators=150, max_samples=5000):
    from sklearn.ensemble import RandomForestClassifier
    
    # Liste toutes les images RX
    img_paths = sorted([p for p in glob(os.path.join(img_dir, "*.*")) if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    X_list, y_list = [], []

    print(f"--- Analyse des dossiers ---")
    print(f"Images RX trouvées : {len(img_paths)}")

    for p in img_paths:
        img_name = os.path.basename(p)
        base_name = os.path.splitext(img_name)[0]
        
        # LOGIQUE FLEXIBLE : cherche base_name + "_label" + n'importe quelle extension
        label_pattern = os.path.join(lbl_dir, f"{base_name}_label.*")
        found_labels = glob(label_pattern)

        if not found_labels:
            print(f"⚠️ Ignoré : Pas de label trouvé pour {img_name} (Pattern: {base_name}_label.*)")
            continue

        lp = found_labels[0]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        lbl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        
        if img is None or lbl is None: 
            print(f"❌ Erreur lecture : {img_name}")
            continue

        print(f"✅ Association : {img_name} <--> {os.path.basename(lp)}")

        feat = compute_features(img)
        feat_flat = feat.reshape(-1, feat.shape[-1])
        lbl_flat = (lbl.flatten() > 127).astype(np.uint8)

        # Sous-échantillonnage
        indices = np.random.choice(len(lbl_flat), min(len(lbl_flat), max_samples), replace=False)
        X_list.append(feat_flat[indices])
        y_list.append(lbl_flat[indices])

    if not X_list:
        print("❌ AUCUNE DONNÉE. Vérifiez que vos labels finissent bien par '_label' (ex: image1_label.png)")
        return
    
    X, y = np.vstack(X_list), np.concatenate(y_list)
    print(f"🚀 Entraînement sur {X.shape[0]} points...")
    
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight="balanced", max_depth=None, random_state=42)
    clf.fit(X, y)
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "model_rx.joblib")
    joblib.dump(clf, save_path)
    print(f"💾 MODÈLE SAUVEGARDÉ : {save_path}")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--images-dir", required=True)
    train_p.add_argument("--labels-dir", required=True)
    train_p.add_argument("--models-dir", required=True)
    train_p.add_argument("--n-estimators", type=int, default=150)
    train_p.add_argument("--max-samples-per-image", type=int, default=5000)

    args = parser.parse_args()
    if args.cmd == "train":
        train_model(args.images_dir, args.labels_dir, args.models_dir, args.n_estimators, args.max_samples_per_image)
