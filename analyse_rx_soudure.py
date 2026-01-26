# analyse_rx_soudure.py
# Auteur : Pierre (assisté par M365 Copilot)
# Objet : % de couverture de soudure dans la zone utile (VERT − trous NOIR)
#         • Masque unique : ALIGNEMENT MANUEL (translation + rotation + échelle isotrope), pas d'automatique
#         • Masques individuels : alignement auto par GRILLE 4×4 (cercles) + RANSAC 2‑points; fallback contours
#         • Optimisations : pré-calcul parallèle + cache des prédictions, preview réduite, CSV global unique
#         • Overlay : JAUNE = brasure présente, ROUGE = manque
#         • CSV : taux de manque (missing_ratio / missing_pct)

import os
import cv2
import math
import numpy as np
import pandas as pd
import argparse
import hashlib
from datetime import datetime
from glob import glob
from joblib import Parallel, delayed  # Option B: parallélisation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Évite la sur-threading OpenCV
try:
    cv2.setNumThreads(1)
except Exception:
    pass

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".bpm")

# =========================
# Utils
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(d: str):
    if not os.path.isdir(d):
        return []
    return sorted(
        [os.path.join(d, n) for n in os.listdir(d) if n.lower().endswith(VALID_EXTS)]
    )

def load_gray(path, contrast_limit=2.0) -> np.ndarray:
    """
    Charge une image et applique un contraste adaptatif (CLAHE).
    """
    # Gestion du flux Streamlit ou du chemin texte
    if hasattr(path, 'read'):
        file_bytes = np.asarray(bytearray(path.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        path.seek(0) # Important pour Streamlit
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    # Conversion en gris si nécessaire
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Passage en 8 bits
    if img.dtype == np.uint16:
        mn, mx = int(img.min()), int(img.max())
        img = ((img.astype(np.float32) - mn) / (mx - mn + 1e-5) * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- LE PATCH CONTRASTE RÉGLABLE ---
    if contrast_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=contrast_limit, tileGridSize=(8,8))
        img = clahe.apply(img)

    return img

def compute_confidence(clf, features):
    """Calcule le score de confiance basé sur les probabilités de l'IA"""
    # Récupère les probabilités pour chaque classe (Soudure vs Fond)
    probs = clf.predict_proba(features)
    # On prend la probabilité la plus haute pour chaque pixel
    max_probs = np.max(probs, axis=1)
    # Retourne la moyenne de certitude sur toute l'image
    return np.mean(max_probs)

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(img)

import math

def compute_features(img: np.ndarray) -> np.ndarray:
    imgf = img.astype(np.float32)
    
    # 1. Intensités (Correction engine. supprimée)
    f_int = imgf / 255.0
    f_clahe = apply_clahe(img) / 255.0 
    
    # 2. Gradients (Scharr pour la précision des bords droits)
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    
    # 3. Contextes (Moyennes glissantes)
    mean3 = cv2.blur(imgf, (3, 3)) / 255.0
    mean15 = cv2.blur(imgf, (15, 15)) / 255.0
    
    # 4. Filtre de continuité verticale (Pour les bandes latérales)
    # Noyau de 15 pixels de haut sur 1 pixel de large
    vertical_kernel = np.ones((15, 1), np.float32) / 15.0
    v_continuity = cv2.filter2D(f_int, -1, vertical_kernel)
    
    # 5. Différence de Gaussiennes (DoG) - Isole les ronds des lignes
    dog = (cv2.GaussianBlur(imgf, (3,3), 0) - cv2.GaussianBlur(imgf, (13,13), 0))
    dog = cv2.normalize(dog, None, 0, 1, cv2.NORM_MINMAX)

    # 6. Gabor (Stabilité des pads)
    g_0 = cv2.normalize(cv2.filter2D(imgf, cv2.CV_32F, cv2.getGaborKernel((15,15), 4.0, 0, 10.0, 0.5, 0)), None, 0, 1, cv2.NORM_MINMAX)
    g_90 = cv2.normalize(cv2.filter2D(imgf, cv2.CV_32F, cv2.getGaborKernel((15,15), 4.0, math.pi/2, 10.0, 0.5, 0)), None, 0, 1, cv2.NORM_MINMAX)

    return np.stack([f_int, f_clahe, mag/mag.max(), dog, mean3, mean15, v_continuity, g_0, g_90], axis=-1)

# =========================
# Labels (scribbles) — tolérant (.png/.jpg/.jpeg)
# =========================
def parse_labels_mask(label_path: str, H: int, W: int) -> np.ndarray:
    lbl = cv2.imread(label_path, cv2.IMREAD_COLOR)
    if lbl is None:
        return None
    if lbl.shape[:2] != (H, W):
        lbl = cv2.resize(lbl, (W, H), interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(lbl, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170,80,80), (180,255,255))
    red_hsv = cv2.bitwise_or(red1, red2)
    yellow_hsv = cv2.inRange(hsv, (18,80,80), (45,255,255))

    b,g,r = cv2.split(lbl)
    red_rgb    = ((r > 160) & (g < 140) & (b < 140)).astype(np.uint8) * 255
    yellow_rgb = ((r > 160) & (g > 160) & (b < 140)).astype(np.uint8) * 255

    red_mask    = cv2.bitwise_or(red_hsv,    red_rgb)
    yellow_mask = cv2.bitwise_or(yellow_hsv, yellow_rgb)

    kernel = np.ones((3,3), np.uint8)
    red_mask    = cv2.morphologyEx(red_mask,    cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    y = np.full((H, W), 255, dtype=np.uint8)
    y[red_mask > 0] = 1
    y[yellow_mask > 0] = 0

    if (y != 255).sum() == 0:
        return None
    return y

# =========================
# Masque -> zone + trous (centres+rayons)
# =========================
def compute_zone_and_holes(inspect_path: str, green_thr: int = 100, black_thr: int = 40):
    insp = cv2.imread(inspect_path, cv2.IMREAD_COLOR)
    if insp is None:
        raise FileNotFoundError(inspect_path)
    b,g,r = cv2.split(insp)
    zone = (g > green_thr).astype(np.uint8)
    holes = ((b < black_thr) & (g < black_thr) & (r < black_thr)).astype(np.uint8)
    holes = (holes & (zone == 1)).astype(np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    zone_use = ((zone == 1) & (holes == 0)).astype(np.uint8)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(holes, 8)
    centers = []
    radii = []
    for lab in range(1, num):
        x, y = centroids[lab]
        area = stats[lab, cv2.CC_STAT_AREA]
        r_est = float(max(1.0, np.sqrt(area/np.pi)))
        centers.append((float(x), float(y)))
        radii.append(r_est)
    centers = np.array(centers, np.float32) if centers else np.empty((0,2), np.float32)
    radii   = np.array(radii,  np.float32) if radii   else np.empty((0,),   np.float32)
    return zone_use, centers, radii

# =========================
# Détection de cercles (Hough) dans une ROI (masques individuels)
# =========================
def detect_circles_in_roi(img: np.ndarray, roi_rect, r_hint: float = None):
    H, W = img.shape
    x0,y0,x1,y1 = roi_rect
    x0 = max(0, min(W-1, x0)); x1 = max(0, min(W-1, x1))
    y0 = max(0, min(H-1, y0)); y1 = max(0, min(H-1, y1))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0,2), np.float32), None

    roi = img[y0:y1+1, x0:x1+1]
    eq = apply_clahe(roi)

    if r_hint is None or not np.isfinite(r_hint):
        min_r, max_r = 4, 30
        min_dist = 14
    else:
        min_r = max(3, int(round(r_hint*0.6)))
        max_r = max(min_r+2, int(round(r_hint*1.6)))
        min_dist = max(8, int(round(r_hint*2.2)))

    pts = []
    radii = []
    for inv in [False, True]:
        src = 255 - eq if inv else eq
        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
                                   param1=120, param2=18, minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            for (x,y,r) in circles[0]:
                pts.append((float(x), float(y))); radii.append(float(r))

    if not pts:
        return np.empty((0,2), np.float32), None

    pts = np.array(pts, np.float32)
    pts[:,0] += x0; pts[:,1] += y0  # coords globales
    r_med = float(np.median(radii)) if len(radii)>0 else None
    return pts, r_med

# =========================
# Grille 4×4 (ordre canonique) + Similarité (S+R+T) — masques individuels
# =========================
def order_mask_grid_4x4(pts: np.ndarray) -> np.ndarray:
    if pts is None or pts.shape[0] < 16:
        return None
    P = pts.copy()
    P = P[np.argsort(P[:,1])]
    rows = np.array_split(P, 4)
    rows = [r[np.argsort(r[:,0])] for r in rows]
    G = np.vstack(rows)
    return G.reshape(4,4,2)

def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]
    mu_src = src.mean(axis=0); mu_dst = dst.mean(axis=0)
    src_c = src - mu_src; dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    var_src = (src_c**2).sum() / n
    s = (S.sum()) / var_src if var_src > 1e-8 else 1.0
    t = mu_dst - s * (R @ mu_src)
    M = np.zeros((2,3), np.float32)
    M[:2,:2] = s * R; M[:,2] = t
    return M

def similarity_from_two_pairs(a: np.ndarray, b: np.ndarray, A: np.ndarray, B: np.ndarray):
    va = b - a; vA = B - A
    da = np.linalg.norm(va); dA = np.linalg.norm(vA)
    if da < 1e-6 or dA < 1e-6: return None
    s = dA / da
    ang_a = math.atan2(va[1], va[0])
    ang_A = math.atan2(vA[1], vA[0])
    th = ang_A - ang_a
    c, si = math.cos(th), math.sin(th)
    R = np.array([[c, -si],[si, c]], dtype=np.float32)
    t = A - s * (R @ a)
    M = np.zeros((2,3), np.float32)
    M[:2,:2] = s * R; M[:,2] = t
    return M

def build_image_grid_4x4(candidates: np.ndarray) -> np.ndarray:
    if candidates is None or candidates.shape[0] < 16:
        return None
    Z = candidates[:,1].astype(np.float32).reshape(-1,1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    K = 4
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    order_rows = np.argsort(centers[:,0])
    rows_pts = []
    for rid in order_rows:
        rpts = candidates[labels.ravel()==rid]
        if rpts.shape[0] < 4: return None
        rpts = rpts[np.argsort(rpts[:,0])]
        if rpts.shape[0] > 4:
            idx = np.linspace(0, rpts.shape[0]-1, 4).round().astype(int)
            rpts = rpts[idx]
        rows_pts.append(rpts[:4])
    G = np.stack(rows_pts, axis=0)
    return G

def grid_align_similarity_ransac(zone: np.ndarray,
                                 holes_mask_xy: np.ndarray,
                                 img: np.ndarray,
                                 r_hint: float = None,
                                 max_iter: int = 300,
                                 inlier_tau_ratio: float = 0.45) -> np.ndarray:
    """Alignement auto (masques individuels) : grille 4×4 + RANSAC 2-points, puis Umeyama sur inliers."""
    H, W = img.shape
    zone_bin = (zone > 0).astype(np.uint8)
    if holes_mask_xy is None or holes_mask_xy.shape[0] < 16:
        return zone_bin

    ys, xs = np.where(zone_bin > 0)
    if ys.size == 0:
        return zone_bin
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    pad = 6
    det_xy, _ = detect_circles_in_roi(img, (x0-pad, y0-pad, x1+pad, y1+pad), r_hint)
    if det_xy.shape[0] < 8:
        return zone_bin

    Gm = order_mask_grid_4x4(holes_mask_xy)
    if Gm is None:
        return zone_bin
    mask_pts = Gm.reshape(-1,2).astype(np.float32)

    dx = np.linalg.norm(Gm[:,1:,:]-Gm[:,:-1,:], axis=2).flatten()
    dy = np.linalg.norm(Gm[1:,:,:]-Gm[:-1,:,:], axis=2).flatten()
    pitch = float(np.median(np.concatenate([dx,dy]))) if dx.size+dy.size>0 else 10.0
    tau = max(4.0, inlier_tau_ratio * pitch)

    corners = np.array([Gm[0,0], Gm[0,3], Gm[3,0], Gm[3,3]], dtype=np.float32)

    best = (0, 1e9, None)
    rng = np.random.default_rng(42)
    N_det = det_xy.shape[0]
    if N_det < 2:
        return zone_bin

    for _ in range(max_iter):
        i1, i2 = rng.choice(4, size=2, replace=False)
        a, b = corners[i1], corners[i2]
        j1, j2 = rng.choice(N_det, size=2, replace=False)
        A, B = det_xy[j1], det_xy[j2]
        M = similarity_from_two_pairs(a, b, A, B)
        if M is None: continue
        ones = np.ones((mask_pts.shape[0],1), np.float32)
        srcH = np.hstack([mask_pts, ones])
        trans = (M[:,:2] @ srcH.T + M[:,2:3]).T
        d2 = np.sum((trans[:,None,:] - det_xy[None,:,:])**2, axis=2)
        nn = np.argmin(d2, axis=1)
        d  = np.sqrt(d2[np.arange(d2.shape[0]), nn])
        inliers = d < tau
        n_inl = int(inliers.sum())
        if n_inl >= 12:
            src_inl = mask_pts[inliers]
            dst_inl = det_xy[nn[inliers]]
            M_ref = umeyama_similarity(src_inl, dst_inl)
            srcH2 = np.hstack([mask_pts, np.ones((mask_pts.shape[0],1), np.float32)])
            trans2 = (M_ref[:2,:2] @ srcH2.T + M_ref[:,2:3]).T
            d2r = np.sum((trans2[:,None,:] - det_xy[None,:,:])**2, axis=2)
            dr  = np.sqrt(np.min(d2r, axis=1))
            mean_err = float(np.mean(dr[inliers]))
            if (n_inl > best[0]) or (n_inl == best[0] and mean_err < best[1]):
                best = (n_inl, mean_err, M_ref)

    if best[2] is None:
        return zone_bin
    zone_u8 = (zone_bin > 0).astype(np.uint8) * 255
    warped = cv2.warpAffine(zone_u8, best[2], (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return (warped > 0).astype(np.uint8)

# =========================
# Fallback léger par contours (secours auto pour masques individuels)
# =========================
def slight_adjust_zone_edges(zone: np.ndarray,
                             img: np.ndarray,
                             shift_max: int = 6, shift_step: int = 2,
                             scale_min: float = 0.97, scale_max: float = 1.03, scale_step: float = 0.01) -> np.ndarray:
    H, W = img.shape
    zone_u8 = (zone > 0).astype(np.uint8) * 255
    zone_edge = cv2.morphologyEx(zone_u8, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    edges_img = cv2.Canny(apply_clahe(img), 60, 120)
    ys, xs = np.where(zone > 0)
    if ys.size == 0:
        return (zone > 0).astype(np.uint8)
    cx, cy = float(xs.mean()), float(ys.mean())
    def warp_with(dx, dy, s):
        M = np.array([[s, 0, (1 - s) * cx + dx],
                      [0, s, (1 - s) * cy + dy]], dtype=np.float32)
        we = cv2.warpAffine(zone_edge, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        wz = cv2.warpAffine(zone_u8,   M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        return we, wz
    best = (-1, None)
    scales = np.arange(scale_min, scale_max + 1e-9, scale_step)
    shifts = range(-shift_max, shift_max + 1, shift_step)
    for s in scales:
        for dy in shifts:
            for dx in shifts:
                we, wz = warp_with(dx, dy, s)
                score = int(((we > 0) & (edges_img > 0)).sum())
                if score > best[0]:
                    best = (score, wz)
    if best[1] is None:
        return (zone > 0).astype(np.uint8)
    return (best[1] > 0).astype(np.uint8)

# =========================
# MANUEL (masque unique) : invite CLI (dx, dy, rot, scale) + preview réduite
# =========================
def compose_similarity(s: float, theta_deg: float, tx: float, ty: float, cx: float, cy: float) -> np.ndarray:
    th = math.radians(theta_deg)
    c, si = math.cos(th), math.sin(th)
    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], np.float32)
    S  = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float32)
    R  = np.array([[c,-si,0],[si,c,0],[0,0,1]], np.float32)
    T2 = np.array([[1,0,cx+tx],[0,1,cy+ty],[0,0,1]], np.float32)
    M3 = T2 @ R @ S @ T1
    return M3[:2,:]

def _save_preview_resized(overlay: np.ndarray, path: str, max_w: int = 1024):
    H0, W0 = overlay.shape[:2]
    if W0 > max_w:
        r = max_w / float(W0)
        overlay = cv2.resize(overlay, (max_w, int(H0*r)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, overlay)

def manual_align_single(img: np.ndarray,
                        zone: np.ndarray,
                        pred_bin: np.ndarray,
                        base: str,
                        out_dir: str) -> np.ndarray:
    """
    Invite utilisateur :
      w/a/s/d (translation) | z/x (échelle -/+) | q/e (rotation -/+)
      dx= dy= s= rot= | p (print) | r (reset) | ok (valider) | skip (passer) | h (aide)
    Preview réduite enregistrée après chaque action dans '.../debug/<base>_manual_preview.png'
    """
    H, W = img.shape
    zone_u8 = (zone > 0).astype(np.uint8) * 255
    ys, xs = np.where(zone > 0)
    if ys.size == 0:
        return (zone > 0).astype(np.uint8)
    cx, cy = float(xs.mean()), float(ys.mean())

    s = 1.0; rot = 0.0; tx = 0.0; ty = 0.0
    dbg_dir = os.path.join(out_dir, "debug"); ensure_dir(dbg_dir)
    preview = os.path.join(dbg_dir, f"{base}_manual_preview.png")

    def render():
        M = compose_similarity(s, rot, tx, ty, cx, cy)
        wz = cv2.warpAffine(zone_u8, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        present_mask = ((wz > 0) & (pred_bin > 0))
        missing_mask = ((wz > 0) & (pred_bin == 0))
        yellow_layer = np.zeros_like(overlay); yellow_layer[present_mask] = (0,255,255)
        red_layer    = np.zeros_like(overlay); red_layer[missing_mask]  = (0,0,255)
        overlay = cv2.addWeighted(overlay, 1.0, yellow_layer, 0.45, 0.0)
        overlay = cv2.addWeighted(overlay, 1.0, red_layer,    0.60, 0.0)
        _save_preview_resized(overlay, preview, max_w=1024)
        return wz

    print("\n=== MODE MANUEL (masque unique) ===")
    print("Prévisualisation (réduite) mise à jour :", preview)
    print("Commandes : w/a/s/d | z/x | q/e | dx= dy= s= rot= | p | r | ok | skip | h")
    wz = render()

    nudge = 2.0; s_step = 0.005; r_step = 0.5

    while True:
        try:
            cmd = input(">> ").strip().lower()
        except EOFError:
            print("[MANUAL] Input interrompu → skip.")
            return (zone > 0).astype(np.uint8)

        if cmd in ("ok","o","y","yes"): return (wz > 0).astype(np.uint8)
        if cmd in ("skip","pass","n","no"): return (zone > 0).astype(np.uint8)
        if cmd in ("h","help","?"):
            print("Exemples : 'dx=3 dy=-2', 's=1.012', 'rot=0.8', 'w', 'a', 'z', 'x', 'ok'"); continue
        if cmd in ("p","print"):
            print(f"state: s={s:.5f}, rot={rot:.2f}, tx={tx:.2f}, ty={ty:.2f}"); continue
        if cmd in ("r","reset"):
            s, rot, tx, ty = 1.0, 0.0, 0.0, 0.0; wz = render(); print("reset."); continue

        if cmd == "w": ty -= nudge; wz = render(); continue
        if cmd == "s": ty += nudge; wz = render(); continue
        if cmd == "a": tx -= nudge; wz = render(); continue
        if cmd == "d": tx += nudge; wz = render(); continue
        if cmd == "z": s  = max(0.5, s - s_step); wz = render(); continue
        if cmd == "x": s  = min(1.5, s + s_step); wz = render(); continue
        if cmd == "q": rot -= r_step; wz = render(); continue
        if cmd == "e": rot += r_step; wz = render(); continue

        try:
            parts = cmd.replace(",", " ").split(); changed = False
            for p in parts:
                if "=" in p:
                    k,v = p.split("=",1)
                    if k == "dx": tx += float(v); changed = True
                    elif k == "dy": ty += float(v); changed = True
                    elif k == "s":  s  = float(v);  changed = True
                    elif k == "rot": rot = float(v); changed = True
                elif p.startswith("nudge="):
                    nudge = float(p.split("=",1)[1]); print("nudge:", nudge)
                elif p.startswith("s_step="):
                    s_step = float(p.split("=",1)[1]); print("s_step:", s_step)
                elif p.startswith("r_step="):
                    r_step = float(p.split("=",1)[1]); print("r_step:", r_step)
            if changed:
                wz = render(); continue
            print("Commande non reconnue. Tape 'h' pour l'aide.")
        except Exception as e:
            print("Erreur parsing commande :", e)

# =========================
# TRAIN
# =========================
def train_model(images_dir: str, labels_dir: str, models_dir: str = "./MyDrive/OBC_mainboard/models",
                n_estimators: int = 500, max_samples_per_image: int = 40000):
    ensure_dir(models_dir)
    ims = list_images(images_dir)
    if not ims:
        raise FileNotFoundError(f"Aucune image trouvée dans {images_dir}")
    Xs, ys = [], []
    for p in ims:
        base = os.path.splitext(os.path.basename(p))[0]
        img = load_gray(p); H, W = img.shape

        label_candidates = [
            os.path.join(labels_dir, f"{base}_label.png"),
            os.path.join(labels_dir, f"{base}_label.jpg"),
            os.path.join(labels_dir, f"{base}_label.jpeg"),
        ]
        lab = next((q for q in label_candidates if os.path.exists(q)), None)
        if lab is None: continue

        ymask = parse_labels_mask(lab, H, W)
        if ymask is None: continue

        yy, xx = np.where(ymask != 255)
        if yy.size == 0: continue

        F = compute_features(img)
        X = F[yy, xx, :]
        y = ymask[yy, xx]

        if len(y) > max_samples_per_image:
            idx = np.random.choice(len(y), max_samples_per_image, replace=False)
            X = X[idx]; y = y[idx]

        Xs.append(X); ys.append(y)

    if not Xs:
        raise RuntimeError("Aucun pixel labelisé trouvé. Ajoute des scribbles rouge/jaune dans ./MyDrive/OBC_mainboard/labels.")

    X = np.vstack(Xs); y = np.concatenate(ys)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight="balanced", random_state=42, max_depth=10, min_samples_leaf=100).fit(X, y)

    yp = clf.predict(X)
    print("=== Rapport d'entraînement (indicatif) ===")
    print(classification_report(y, yp, target_names=["background", "solder"]))
    print("Confusion matrix:"); print(confusion_matrix(y, yp))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp = os.path.join(models_dir, f"rf_solder_{ts}.joblib")
    joblib.dump(clf, mp)
    pd.Series({"model_path": mp}).to_json(os.path.join(models_dir, f"rf_solder_{ts}.json"))
    print(f"[OK] Modèle sauvegardé : {mp}")
    return mp

# =========================
# PRÉ-CALCUL PARALLÈLE + CACHE DES PRÉDICTIONS
# =========================
def _hash_model_path(model_path:str) -> str:
    return hashlib.sha1(model_path.encode("utf-8")).hexdigest()[:10]

def _predict_one(p, clf, cache_dir):
    base = os.path.splitext(os.path.basename(p))[0]
    img = load_gray(p); H, W = img.shape
    cache_path = os.path.join(cache_dir, f"{base}_pred.npy")
    if os.path.exists(cache_path):
        try:
            pred_bin = np.load(cache_path)
            if pred_bin.shape == (H, W):
                return base, pred_bin
        except Exception:
            pass
    # calcul unique
    F = compute_features(img)
    pred = clf.predict(F.reshape(-1, F.shape[-1])).reshape(H, W).astype(np.uint8)
    pred_bin = (pred == 1).astype(np.uint8) * 255
    np.save(cache_path, pred_bin)
    return base, pred_bin

def precompute_predictions_parallel(images: list, clf, cache_dir: str, n_jobs: int = -1) -> dict:
    ensure_dir(cache_dir)
    items = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_predict_one)(p, clf, cache_dir) for p in images)
    return {k: v for k, v in items}

# =========================
# INFER
# =========================
def infer_and_save(images_dir: str, masks_dir: str, model_path: str, output_dir: str = "./MyDrive/OBC_mainboard/resultats",
                   green_threshold: int = 100, black_threshold: int = 40,
                   single_mask: bool = False, single_mask_name: str = "zone_inspect.png",
                   manual_single: bool = True,      # masque unique : manuel par défaut
                   allow_auto_adjust: bool = True,  # masques individuels : auto autorisé
                   shift_max: int = 5, shift_step: int = 1,
                   scale_min: float = 0.96, scale_max: float = 1.04, scale_step: float = 0.004,
                   debug: bool = False):
    ensure_dir(output_dir)
    if debug: ensure_dir(os.path.join(output_dir, "debug"))

    # model_path peut être un dossier
    if os.path.isdir(model_path):
        c = sorted(glob(os.path.join(model_path, "*.joblib")), key=os.path.getmtime)
        if not c: raise FileNotFoundError(f"Aucun modèle .joblib dans {model_path}")
        print(f"[INFO] Modèle auto-sélectionné : {c[-1]}")
        model_path = c[-1]
    clf = joblib.load(model_path)

    ims = list_images(images_dir)
    if not ims: raise FileNotFoundError(f"Aucune image trouvée dans {images_dir}")

    # ====== PRÉ-CALCUL PARALLÈLE & CACHE PREDICTIONS ======
    model_tag = _hash_model_path(model_path)
    pred_cache_dir = os.path.join(output_dir, f"_pred_cache_{model_tag}")
    preds_map = precompute_predictions_parallel(ims, clf, pred_cache_dir, n_jobs=-1)
    # ======================================================

    # Masque unique (pré-chargé)
    zone_single = None
    if single_mask:
        ip = os.path.join(masks_dir, single_mask_name)
        z, _, _ = compute_zone_and_holes(ip, green_threshold, black_threshold)
        zone_single = z

    rows = []
    for p in ims:
        base = os.path.splitext(os.path.basename(p))[0]
        img = load_gray(p); H, W = img.shape

        # Prédiction depuis cache
        pred_bin = preds_map[base]

        # Zone utile (et trous si masques individuels)
        holes_xy = None; holes_r = None
        if single_mask:
            if zone_single is None: raise FileNotFoundError("Masque unique introuvable.")
            zone = zone_single
        else:
            ip = os.path.join(masks_dir, f"{base}_inspect.png")
            if not os.path.exists(ip):
                print(f"[WARN] Pas de masque pour {base} → image ignorée."); continue
            z, hxy, hr = compute_zone_and_holes(ip, green_threshold, black_threshold)
            zone, holes_xy, holes_r = z, hxy, hr

        # Resize si besoin (et échelle des centres)
        if zone.shape != (H, W):
            Hz, Wz = zone.shape
            zone = cv2.resize(zone, (W, H), interpolation=cv2.INTER_NEAREST)
            if holes_xy is not None and holes_xy.shape[0] > 0:
                sx = W / Wz; sy = H / Hz
                holes_xy = holes_xy.copy(); holes_xy[:,0] *= sx; holes_xy[:,1] *= sy

        zone_adj = (zone > 0).astype(np.uint8)

        # ---------- Masque unique : ALIGNEMENT MANUEL ----------
        if single_mask:
            if manual_single:
                zone_adj = manual_align_single(img, zone_adj, pred_bin, base, output_dir)
            # sinon : identité (pas d'auto pour le masque unique)

        # ---------- Masques individuels : AUTO (grille) + fallback ----------
        else:
            if allow_auto_adjust and holes_xy is not None and holes_xy.shape[0] >= 16:
                r_hint = float(np.median(holes_r)) if holes_r is not None and holes_r.size > 0 else None
                zone_try = grid_align_similarity_ransac(zone_adj, holes_xy, img, r_hint=r_hint)
                if zone_try.sum() > 0:
                    zone_adj = zone_try
                else:
                    zone_adj = slight_adjust_zone_edges(zone_adj, img,
                                                        shift_max=max(3, shift_max),
                                                        shift_step=max(1, shift_step),
                                                        scale_min=max(0.98, scale_min),
                                                        scale_max=min(1.02, scale_max),
                                                        scale_step=max(0.005, scale_step))
            # sinon : identité

        # -------- Métriques : taux de manque --------
        zone_area = int(zone_adj.sum())
        solder_px = int(((pred_bin > 0) & (zone_adj > 0)).sum())
        missing_px    = zone_area - solder_px
        missing_ratio = (missing_px / zone_area) if zone_area > 0 else 0.0
        missing_pct   = missing_ratio * 100.0

        # -------- Overlay : JAUNE = PRESENT, ROUGE = MANQUE --------
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        present_mask = ((zone_adj > 0) & (pred_bin > 0))
        missing_mask = ((zone_adj > 0) & (pred_bin == 0))
        yellow_layer = np.zeros_like(overlay, dtype=np.uint8)
        red_layer    = np.zeros_like(overlay, dtype=np.uint8)
        yellow_layer[present_mask] = (0, 255, 255)
        red_layer[missing_mask]    = (0,   0, 255)
        overlay = cv2.addWeighted(overlay, 1.0, yellow_layer, 0.45, 0.0)
        overlay = cv2.addWeighted(overlay, 1.0, red_layer,    0.60, 0.0)

        # Sauvegardes images
        cv2.imwrite(os.path.join(output_dir, f"{base}_pred_solder.png"), pred_bin)
        cv2.imwrite(os.path.join(output_dir, f"{base}_overlay.png"), overlay)

        if debug:
            dbg = overlay.copy()
            txt = f"{'manual-single' if single_mask and manual_single else ('auto-per-image' if not single_mask else 'identity-single')}"
            cv2.putText(dbg, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            ensure_dir(os.path.join(output_dir, "debug"))
            cv2.imwrite(os.path.join(output_dir, "debug", f"{base}_debug.png"), dbg)

        rows.append({
            "image": base,
            "zone_area_px": zone_area,
            "solder_area_px": solder_px,
            "missing_area_px": missing_px,
            "missing_ratio": round(missing_ratio, 6),
            "missing_pct": round(missing_pct, 4),
            "single_mask_used": bool(single_mask),
            "manual_single": bool(single_mask and manual_single)
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "metrics_global.csv"), index=False)
        print(f"[OK] Résultats sauvegardés → {output_dir}")
    else:
        print("[INFO] Aucune image traitée (vérifie noms et masques).")

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="RX : % de soudure dans zone utile. Masque unique (manuel) ou masques individuels (auto)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    default_root   = "./MyDrive/OBC_mainboard"
    default_images = os.path.join(default_root, "rx_images")
    default_labels = os.path.join(default_root, "labels")
    default_masks  = os.path.join(default_root, "masks")
    default_models = os.path.join(default_root, "models")
    default_out    = os.path.join(default_root, "resultats")

    # train
    t = sub.add_parser("train", help="Entraîner le modèle (scribbles rouge=1 / jaune=0).")
    t.add_argument("--images-dir", type=str, default=default_images)
    t.add_argument("--labels-dir", type=str, default=default_labels)
    t.add_argument("--models-dir", type=str, default=default_models)
    t.add_argument("--n-estimators", type=int, default=300)
    t.add_argument("--max-samples-per-image", type=int, default=40000)

    # infer
    i = sub.add_parser("infer", help="Inférence + métriques.")
    i.add_argument("--images-dir", type=str, default=default_images)
    i.add_argument("--masks-dir",  type=str, default=default_masks)
    i.add_argument("--model-path", type=str, required=True)  # peut être un dossier
    i.add_argument("--output-dir", type=str, default=default_out)
    i.add_argument("--green-threshold", type=int, default=100)
    i.add_argument("--black-threshold", type=int, default=40)

    # masque unique (manuel)
    i.add_argument("--single-mask", action="store_true")
    i.add_argument("--single-mask-name", type=str, default="zone_inspect.png")
    i.add_argument("--manual-single", action="store_true", default=True,
                   help="Masque unique : alignement manuel (translation + rotation + échelle).")

    # masques individuels — auto (grille + fallback)
    i.add_argument("--allow-auto-adjust", action="store_true", default=True,
                   help="Masques individuels : autoriser l'alignement auto (grille) + fallback contours.")

    # fallback contours (seulement si auto)
    i.add_argument("--shift-max",  type=int,   default=5)
    i.add_argument("--shift-step", type=int,   default=1)
    i.add_argument("--scale-min",  type=float, default=0.96)
    i.add_argument("--scale-max",  type=float, default=1.04)
    i.add_argument("--scale-step", type=float, default=0.004)

    i.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.cmd == "train":
        train_model(args.images_dir, args.labels_dir, args.models_dir,
                    args.n_estimators, args.max_samples_per_image)
    else:
        infer_and_save(args.images_dir, args.masks_dir, args.model_path, args.output_dir,
                       args.green_threshold, args.black_threshold,
                       args.single_mask, args.single_mask_name,
                       args.manual_single,
                       args.allow_auto_adjust,
                       args.shift_max, args.shift_step,
                       args.scale_min, args.scale_max, args.scale_step,
                       args.debug)

if __name__ == "__main__":
    main()
