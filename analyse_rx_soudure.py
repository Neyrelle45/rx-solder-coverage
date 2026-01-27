import os
import cv2
import math
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from glob import glob

# Optimisation OpenCV
try:
    cv2.setNumThreads(1)
except:
    pass

# =========================
# UTILS DE BASE
# =========================
def load_gray(p, contrast_limit=2.0):
    """Charge une image en gris avec un CLAHE optionnel."""
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    if contrast_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=contrast_limit, tileGridSize=(8,8))
        img = clahe.apply(img)
    return img

def apply_clahe(img):
    """Applique un CLAHE standard (utilisé dans compute_features)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def compose_similarity(scale, rotation_deg, tx, ty, cx, cy):
    """Génère la matrice de transformation pour l'alignement manuel."""
    angle = math.radians(rotation_deg)
    a = scale * math.cos(angle)
    b = scale * math.sin(angle)
    M = np.array([
        [a, b, (1-a)*cx - b*cy + tx],
        [-b, a, b*cx + (1-a)*cy + ty]
    ], dtype=np.float32)
    return M

# =========================
# CŒUR IA (PRÉSERVÉ)
# =========================
def compute_features(img: np.ndarray) -> np.ndarray:
    """Extractions des caractéristiques pour le Random Forest."""
    imgf = img.astype(np.float32)
    f_int = imgf / 255.0
    f_clahe = apply_clahe(img).astype(np.float32) / 255.0

    # Gradients Scharr (Précision bords)
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 1, cv2.NORM_MINMAX)

    # Directionnalité (Anisotropie pour différencier bords/voids)
    edge_dir = np.abs(gx) / (np.abs(gy) + 5.0)
    edge_dir = cv2.normalize(np.clip(edge_dir, 0, 10), None, 0, 1, cv2.NORM_MINMAX)

    # Statistiques locales
    mean3 = cv2.blur(imgf, (3, 3)) / 255.0
    mean21 = cv2.blur(imgf, (21, 21)) / 255.0
    
    # Filtre de continuité verticale (Bandes latérales)
    v_kernel = np.ones((25, 1), np.float32) / 25.0
    v_line = cv2.filter2D(f_int, -1, v_kernel)

    # Différence de Gaussiennes (DoG)
    dog = cv2.normalize(cv2.GaussianBlur(imgf,(3,3),0)-cv2.GaussianBlur(imgf,(15,15),0), None, 0, 1, cv2.NORM_MINMAX)

    # Gabor
    g_v = cv2.normalize(cv2.filter2D(imgf, cv2.CV_32F, cv2.getGaborKernel((15,15), 4.0, math.pi/2, 10.0, 0.5, 0)), None, 0, 1, cv2.NORM_MINMAX)
    g_h = cv2.normalize(cv2.filter2D(imgf, cv2.CV_32F, cv2.getGaborKernel((15,15), 4.0, 0, 10.0, 0.5, 0)), None, 0, 1, cv2.NORM_MINMAX)

    return np.stack([f_int, f_clahe, mag, edge_dir, dog, v_line, mean3, mean21, g_v, g_h], axis=-1)

# =========================
# ENTRAÎNEMENT
# =========================
def train_model(img_dir, lbl_dir, out_dir, n_estimators=150, max_samples=5000):
    from sklearn.ensemble import RandomForestClassifier
    
    img_paths = sorted(glob(os.path.join(img_dir, "*.*")))
    X_list, y_list = [], []

    for p in img_paths:
        name = os.path.basename(p)
        lp = os.path.join(lbl_dir, name)
        if not os.path.exists(lp): continue

        img = load_gray(p)
        lbl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if img is None or lbl is None: continue

        feat = compute_features(img)
        h, w, c = feat.shape
        feat_flat = feat.reshape(-1, c)
        lbl_flat = (lbl.flatten() > 127).astype(np.uint8)

        # Sous-échantillonnage pour éviter l'explosion mémoire
        indices = np.random.choice(len(lbl_flat), min(len(lbl_flat), max_samples), replace=False)
        X_list.append(feat_flat[indices])
        y_list.append(lbl_flat[indices])

    if not X_list: return
    
    X, y = np.vstack(X_list), np.concatenate(y_list)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight="balanced", max_depth=None)
    clf.fit(X, y)
    
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(out_dir, "model_rx.joblib"))
    print(f"Modèle sauvegardé dans {out_dir}")

# =========================
# INTERFACE CLI MINIMALE
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--images-dir", required=True)
    train_p.add_argument("--labels-dir", required=True)
    train_p.add_argument("--models-dir", default="./models")
    train_p.add_argument("--n-estimators", type=int, default=150)
    train_p.add_argument("--max-samples-per-image", type=int, default=5000)

    args = parser.parse_args()
    if args.cmd == "train":
        train_model(args.images_dir, args.labels_dir, args.models_dir, 
                    args.n_estimators, args.max_samples_per_image)
