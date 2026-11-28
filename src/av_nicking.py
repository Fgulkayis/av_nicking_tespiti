import os
import pickle
import warnings
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, disk
from skimage.filters import frangi

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJE_KOKU, "models")

# Eğitilmiş modelleri bellekte önbelleğe almak için
_MODEL_CACHE = {}


def _load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    cached = _MODEL_CACHE.get(filename)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        _MODEL_CACHE[filename] = (mtime, obj)
        return obj
    except Exception:
        return None


def _rm_small_objects(m, n):
    try:
        return remove_small_objects(m, max_size=n)
    except TypeError:
        return remove_small_objects(m, min_size=n)


def _rm_small_holes(m, n):
    try:
        return remove_small_holes(m, max_size=n)
    except TypeError:
        return remove_small_holes(m, area_threshold=n)


def _imread(path):
    """Windows'ta Türkçe karakterli dosya yollarını güvenle okur."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


# ================================================================
# 1) YÜKSEK ÇÖZÜNÜRLÜKLÜ ÖN İŞLEME VE AYDINLATMA DÜZELTME
# ================================================================
def load_and_preprocess(path, target_width=1200):
    bgr = _imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    orig_h, orig_w = bgr.shape[:2]
    scale = target_width / float(orig_w)
    new_h = int(round(orig_h * scale))
    bgr = cv2.resize(bgr, (target_width, new_h), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # FOV (Görüş Alanı) maskesi: kırmızı ve yeşil kanaldan arka planı ayır
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    fov = ((red > 18) | (green > 15)).astype(np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    fov = cv2.morphologyEx(fov, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    fov = cv2.erode(fov, np.ones((15, 15), np.uint8))  # kenar parlamasını at

    # Yeşil kanalda CLAHE + arka plan aydınlatma düzeltmesi
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enh = clahe.apply(green)
    enh = cv2.medianBlur(enh, 3)

    return rgb, enh, fov.astype(bool)


# ================================================================
# 2) HİBRİT DAMAR SEGMENTASYONU (ÇOK ÖLÇEKLİ TOP-HAT + FRANGI)
# ================================================================
def compute_vesselness(enh, fov):
    """
    İnce ve sıkışmış (AV nicking) damarların kesişim yerinde kopmasını önlemek için
    küçük/büyük ölçekli Black Top-Hat ile Frangi (Hessian tübülerlik) yanıtını birleştirir.
    """
    th = np.zeros(enh.shape, np.float32)
    # r=3 ve r=5 ince/ezilmiş damarları yakalar; r=9,13,17 ana damarları yakalar
    for r in (3, 5, 8, 12, 16):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        resp = cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        th = np.maximum(th, resp)
    th[~fov] = 0
    th = cv2.GaussianBlur(th, (3, 3), 0)
    if th[fov].size > 0:
        p99 = np.percentile(th[fov], 99.5) + 1e-5
        th_norm = np.clip(th / p99, 0.0, 1.0)
    else:
        th_norm = th

    # Frangi çok ölçekli damar filtresi (yarıya küçültülmüş görüntüde hızlı hesapla)
    h, w = enh.shape
    small = cv2.resize(enh, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    try:
        fr_small = frangi(small, sigmas=(1.0, 2.0, 3.5), black_ridges=True)
        fr = cv2.resize(fr_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        fr[~fov] = 0
        if fr[fov].size > 0 and np.percentile(fr[fov], 99) > 0:
            fr_norm = np.clip(fr / (np.percentile(fr[fov], 99) + 1e-6), 0.0, 1.0)
        else:
            fr_norm = np.zeros_like(th_norm)
    except Exception:
        fr_norm = np.zeros_like(th_norm)

    vesselness = 0.65 * th_norm + 0.35 * fr_norm
    vesselness[~fov] = 0.0
    return vesselness


def segment_vessels(enh, fov, min_size=350, return_vesselness=False):
    vesselness = compute_vesselness(enh, fov)
    vals = vesselness[fov]
    if vals.size == 0:
        empty = np.zeros_like(fov, dtype=bool)
        return (empty, vesselness) if return_vesselness else empty

    # Çift eşikli (Histerezis) segmentasyon: kesişimde zayıflayan damarı ana damara bağlıysa korur
    t_high = np.percentile(vals, 85.5)
    t_low = np.percentile(vals, 79.0)
    strong = (vesselness > t_high) & fov
    weak = (vesselness > t_low) & fov
    mask = ndi.binary_propagation(strong, mask=weak)

    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    mask = _rm_small_objects(mask, min_size)
    mask = ndi.binary_closing(mask, structure=disk(1))
    # Central light reflex (damar ortası parlak yansıma) deliklerini doldur
    mask = _rm_small_holes(mask, 320)
    if return_vesselness:
        return mask, vesselness
    return mask


# ================================================================
# 3) OPTİK DİSK TESPİTİ (PARLAKLIK + DAMAR YOĞUNLUĞU BİRLEŞİMİ)
# ================================================================
def detect_optic_disc(rgb, fov, mask=None):
    """Optik disk: kırmızı+yeşil parlaklığın ve ana damar yoğunluğunun kesiştiği bölge."""
    w = rgb.shape[1]
    lum = (0.5 * rgb[:, :, 0] + 0.5 * rgb[:, :, 1]).astype(np.float32)
    lum[~fov] = 0
    k = int(w * 0.08) | 1
    blur = cv2.GaussianBlur(lum, (k, k), 0)
    if mask is not None and mask.any():
        v_dens = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
        score = blur * (0.5 + 0.5 * (v_dens / (v_dens.max() + 1e-5)))
    else:
        score = blur
    # Kenarlara çok yakın noktaları bastır
    er_fov = cv2.erode(fov.astype(np.uint8), np.ones((int(w * 0.04) | 1,) * 2, np.uint8)) > 0
    if er_fov.any():
        score[~er_fov] = 0
    _, _, _, (x, y) = cv2.minMaxLoc(score)
    radius = int(w * 0.075)
    return (y, x), radius


# ================================================================
# 4) İSKELETLEŞTİRME, BUDAMA VE DAMAR KIVRIMLILIĞI (TORTUOSITY)
# ================================================================
_K = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])


def neighbor_count(skel):
    c = ndi.convolve(skel.astype(np.uint8), _K, mode="constant")
    return np.where(skel, c - 10, 0)


def prune_spurs(skel, length=12, iters=3):
    skel = skel.copy()
    for _ in range(iters):
        nb = neighbor_count(skel)
        junction = (nb >= 3)
        no_j = skel & ~ndi.binary_dilation(junction, structure=np.ones((3, 3)))
        lab, n = ndi.label(no_j, structure=np.ones((3, 3)))
        if n == 0:
            break
        sizes = ndi.sum(no_j, lab, index=np.arange(1, n + 1))
        ends = (nb == 1)
        has_end = ndi.maximum(ends, lab, index=np.arange(1, n + 1))
        dil_j = ndi.binary_dilation(junction, structure=np.ones((5, 5)))
        touches_j = ndi.maximum(dil_j, lab, index=np.arange(1, n + 1))
        remove = np.where((sizes < length) & (has_end > 0) & (touches_j > 0))[0] + 1
        if len(remove) == 0:
            break
        skel[np.isin(lab, remove)] = False
        skel = skeletonize(skel)
    return skel


def compute_vessel_tortuosity(skel, min_len=35):
    """
    Damar segmentlerinin Yay Uzunluğu / Öklid Mesafesi (Arc-Chord Ratio) ve
    yerel eğrilik değişimini ölçerek hipertansif damar kıvrımlılığını hesaplar.
    """
    nb = neighbor_count(skel)
    junction = nb >= 3
    seg_mask = skel & ~ndi.binary_dilation(junction, structure=np.ones((3, 3)))
    lab, n = ndi.label(seg_mask, structure=np.ones((3, 3)))
    if n == 0:
        return 1.05, 0.0, []

    ratios = []
    weights = []
    seg_list = []
    slices = ndi.find_objects(lab)
    for i, slc in enumerate(slices, 1):
        if slc is None:
            continue
        sub = lab[slc] == i
        ys_local, xs_local = np.nonzero(sub)
        arc_len = float(len(ys_local))
        if arc_len < min_len:
            continue
        ys = ys_local + slc[0].start
        xs = xs_local + slc[1].start
        # Uç noktaları bul: segment pikselleri arasında en uzak iki nokta
        pts = np.column_stack([ys, xs]).astype(np.float32)
        # Hızlı uç nokta tahmini: PCA ana ekseni üzerindeki min ve max projeksiyon
        mean_pt = pts.mean(axis=0)
        centered = pts - mean_pt
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, -1]
        proj = centered @ principal
        p_start = pts[np.argmin(proj)]
        p_end = pts[np.argmax(proj)]
        chord_len = float(np.linalg.norm(p_end - p_start))
        if chord_len < 15.0:
            continue
        # 8-komşuluklu iskelette gerçek geometrik yay uzunluğu ve dikey eğrilik sapması
        perp = np.array([-principal[1], principal[0]])
        perp_proj = centered @ perp
        perp_dev = float(np.std(perp_proj) / (chord_len + 1e-5))
        max_arch = float((np.max(perp_proj) - np.min(perp_proj)) / (chord_len + 1e-5))
        # Yay uzunluğu / kiriş uzunluğu yaklaşımı + eğrilik katkısı
        geom_arc = max(chord_len, float(np.sqrt(chord_len**2 + (2.4 * (np.max(perp_proj) - np.min(perp_proj)))**2)))
        ratio = float(np.clip(geom_arc / chord_len + 0.8 * perp_dev, 1.0, 2.5))
        ratios.append(ratio)
        weights.append(arc_len)
        seg_list.append({
            "arc_len": arc_len,
            "chord_len": chord_len,
            "tortuosity": ratio,
            "perp_dev": max_arch,
            "coords": (ys, xs),
        })

    if not ratios:
        return 1.05, 0.0, []
    ratios = np.array(ratios)
    weights = np.array(weights)
    weighted_mean = float(np.average(ratios, weights=weights))
    top_q = float(np.percentile(ratios, 80))
    return weighted_mean, top_q, seg_list


# ================================================================
# 5) ATARDAMAR / TOPLARDAMAR (A/V) ÖZNİTELİK ÇIKARIMI VE MODEL 1
# ================================================================
def prepare_color_channels(rgb, enh):
    """A/V sınıflandırması için arka plan aydınlatmasından arındırılmış renk kanalları hazırlar."""
    rgb_f = rgb.astype(np.float32)
    # Yerel arka plan aydınlatması (geniş Gauss bulanıklaştırma)
    bg_r = cv2.GaussianBlur(rgb_f[:, :, 0], (61, 61), 0) + 1.0
    bg_g = cv2.GaussianBlur(rgb_f[:, :, 1], (61, 61), 0) + 1.0
    bg_b = cv2.GaussianBlur(rgb_f[:, :, 2], (61, 61), 0) + 1.0

    norm_r = rgb_f[:, :, 0] / bg_r
    norm_g = rgb_f[:, :, 1] / bg_g
    norm_b = rgb_f[:, :, 2] / bg_b

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_l = cv2.GaussianBlur(lab[:, :, 0], (61, 61), 0) + 1.0
    norm_l = lab[:, :, 0] / bg_l

    # Central Light Reflex (CLR) haritalarını görüntü başına tek seferde hızlıca hesapla
    k3 = np.ones((5, 5), np.uint8)
    g_u8 = rgb[:, :, 1]
    r_u8 = rgb[:, :, 0]
    min_g = cv2.erode(g_u8, k3).astype(np.float32)
    max_g = cv2.dilate(g_u8, k3).astype(np.float32)
    clr_g_map = (rgb_f[:, :, 1] - min_g) / (max_g - min_g + 1.0)

    min_r = cv2.erode(r_u8, k3).astype(np.float32)
    max_r = cv2.dilate(r_u8, k3).astype(np.float32)
    clr_r_map = (rgb_f[:, :, 0] - min_r) / (max_r - min_r + 1.0)

    return {
        "r": rgb_f[:, :, 0],
        "g": rgb_f[:, :, 1],
        "b": rgb_f[:, :, 2],
        "enh": enh.astype(np.float32),
        "norm_r": norm_r,
        "norm_g": norm_g,
        "norm_b": norm_b,
        "rg_ratio": (rgb_f[:, :, 0] + 1.0) / (rgb_f[:, :, 1] + 1.0),
        "norm_rg_diff": norm_r - norm_g,
        "s": hsv[:, :, 1] / 255.0,
        "v": hsv[:, :, 2] / 255.0,
        "lab_l": norm_l,
        "lab_a": lab[:, :, 1] - 128.0,
        "lab_b": lab[:, :, 2] - 128.0,
        "clr_g": clr_g_map,
        "clr_r": clr_r_map,
    }


def extract_branch_av_features(coords_yx, dist, channels, med_w):
    """
    Bir damar kolu üzerindeki iskelet noktalarından 14 boyutlu optik ve morfolojik
    Atardamar/Toplardamar (A/V) öznitelik vektörü çıkarır:
    - Arka plan normalize R, G, B kontrastları
    - Kırmızı/Yeşil oranı ve farkı (oksijenli hemoglobin imzası)
    - CIE LAB (L*, a*, b*) ve HSV doygunluk değerleri
    - Central Light Reflex (Merkez Yansıma): damar merkezi ile duvarı arasındaki parlaklık farkı
    - Bağıl damar çapı
    """
    h, w = dist.shape
    ys, xs = coords_yx
    valid = (ys >= 2) & (ys < h - 2) & (xs >= 2) & (xs < w - 2)
    ys, xs = ys[valid], xs[valid]
    if len(ys) == 0:
        return np.zeros(14, dtype=np.float32)

    # İskelet (damar merkezi) değerleri
    c_nr = channels["norm_r"][ys, xs]
    c_ng = channels["norm_g"][ys, xs]
    c_nb = channels["norm_b"][ys, xs]
    c_enh = channels["enh"][ys, xs] / 255.0
    c_rg = channels["rg_ratio"][ys, xs]
    c_rg_diff = channels["norm_rg_diff"][ys, xs]
    c_s = channels["s"][ys, xs]
    c_l = channels["lab_l"][ys, xs]
    c_a = channels["lab_a"][ys, xs] / 128.0
    c_b = channels["lab_b"][ys, xs] / 128.0
    widths = 2.0 * dist[ys, xs]
    clr_g = channels["clr_g"][ys, xs]
    clr_r = channels["clr_r"][ys, xs]

    feats = np.array([
        float(np.mean(c_nr)),
        float(np.mean(c_ng)),
        float(np.mean(c_nb)),
        float(np.mean(c_enh)),
        float(np.mean(c_rg)),
        float(np.mean(c_rg_diff)),
        float(np.std(c_ng)),
        float(np.mean(c_s)),
        float(np.mean(c_l)),
        float(np.mean(c_a)),
        float(np.mean(c_b)),
        float(np.mean(clr_g)),
        float(np.mean(clr_r)),
        float(np.median(widths) / max(med_w, 1.0)),
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)


def predict_artery_probabilities_batch(feat_matrix):
    """Birden fazla damar segmentinin Atardamar olasılığını tek seferde (vektörel) hesaplar."""
    if len(feat_matrix) == 0:
        return np.array([], dtype=np.float32)
    X = np.asarray(feat_matrix, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    model_bundle = _load_model("av_classifier.pkl")
    if model_bundle is not None:
        model = model_bundle["model"]
        try:
            return model.predict_proba(X)[:, 1].astype(np.float32)
        except Exception:
            pass

    norm_g = X[:, 1]
    rg_diff = X[:, 5]
    clr_g = X[:, 11]
    rel_w = X[:, 13]
    score = 3.5 * (norm_g - 0.86) + 2.8 * (rg_diff - 0.08) + 0.8 * (clr_g - 0.45) - 0.45 * (rel_w - 1.0)
    return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)


def predict_artery_probability(feat_vec):
    """Tek bir damar kolu için Atardamar olasılığı in [0.0, 1.0] döndürür."""
    return float(predict_artery_probabilities_batch(feat_vec)[0])


def build_av_segmentation_map(rgb, mask, skel, dist, channels, med_w):
    """
    Tüm damar ağını segmentlere ayırıp her segmenti Model 1 ile Atardamar (Kırmızı) veya
    Toplardamar (Mavi) olarak sınıflandırır ve Zone-B AVR (Atardamar/Toplardamar Çap Oranı) hesaplar.
    """
    nb = neighbor_count(skel)
    junction = nb >= 3
    seg_mask = skel & ~ndi.binary_dilation(junction, structure=np.ones((3, 3)))
    lab, n = ndi.label(seg_mask, structure=np.ones((3, 3)))

    artery_skel = np.zeros_like(skel, dtype=bool)
    vein_skel = np.zeros_like(skel, dtype=bool)
    art_widths = []
    vein_widths = []

    seg_coords = []
    seg_feats = []
    seg_w_meds = []
    slices = ndi.find_objects(lab)
    for i, slc in enumerate(slices, 1):
        if slc is None:
            continue
        ys_l, xs_l = np.nonzero(lab[slc] == i)
        if len(ys_l) < 6:
            continue
        ys = ys_l + slc[0].start
        xs = xs_l + slc[1].start
        seg_coords.append((ys, xs))
        seg_feats.append(extract_branch_av_features((ys, xs), dist, channels, med_w))
        seg_w_meds.append(float(2.0 * np.median(dist[ys, xs])))

    if seg_feats:
        p_arts = predict_artery_probabilities_batch(np.array(seg_feats, dtype=np.float32))
        for (ys, xs), p_art, w_med in zip(seg_coords, p_arts, seg_w_meds):
            if p_art >= 0.5:
                artery_skel[ys, xs] = True
                if w_med >= 0.7 * med_w:
                    art_widths.append(w_med)
            else:
                vein_skel[ys, xs] = True
                if w_med >= 0.7 * med_w:
                    vein_widths.append(w_med)

    # İskeletteki A/V etiketlerini en yakın damar maskesi piksellerine yay (Voronoi ataması)
    if artery_skel.any() and vein_skel.any():
        d_art = ndi.distance_transform_edt(~artery_skel)
        d_vein = ndi.distance_transform_edt(~vein_skel)
        artery_mask = mask & (d_art <= d_vein)
        vein_mask = mask & (d_vein < d_art)
    else:
        artery_mask = mask.copy()
        vein_mask = np.zeros_like(mask)

    # Görselleştirme için renkli A/V haritası (Kırmızı = Atardamar, Mavi = Toplardamar)
    av_vis = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    av_vis[artery_mask] = [245, 65, 65]
    av_vis[vein_mask] = [55, 135, 255]

    # AVR (Arteriovenous Ratio - Atardamar / Toplardamar Çap Oranı)
    if art_widths and vein_widths:
        crae = float(np.percentile(art_widths, 70))
        crve = float(np.percentile(vein_widths, 70))
        avr = float(np.clip(crae / max(crve, 1e-5), 0.35, 1.10))
    else:
        crae, crve, avr = float(med_w * 0.85), float(med_w * 1.15), 0.72

    return {
        "artery_mask": artery_mask,
        "vein_mask": vein_mask,
        "av_vis": av_vis,
        "avr": avr,
        "crae_px": crae,
        "crve_px": crve,
    }


# ================================================================
# 6) KESİŞİM TESPİTİ, SUB-PIXEL DİKEY KESİT ÇAP ÖLÇÜMÜ VE SALUS AÇISI
# ================================================================
def find_junction_clusters(skel, merge_radius=8):
    nb = neighbor_count(skel)
    jpts = nb >= 3
    jd = ndi.binary_dilation(jpts, structure=disk(merge_radius))
    lab, n = ndi.label(jd)
    centers = ndi.center_of_mass(jpts, lab, index=np.arange(1, n + 1))
    return [(int(round(r)), int(round(c))) for r, c in centers if not np.isnan(r)]


def ring_branches(skel, center, r_in, r_out):
    """Merkez etrafındaki halkada iskeleti kesen kolların açılarını (radyan) bulur."""
    h, w = skel.shape
    cy, cx = center
    y0, y1 = max(0, cy - r_out - 1), min(h, cy + r_out + 2)
    x0, x1 = max(0, cx - r_out - 1), min(w, cx + r_out + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(yy - cy, xx - cx)
    ring = (d >= r_in) & (d <= r_out)
    sub = skel[y0:y1, x0:x1] & ring
    lab, n = ndi.label(sub, structure=np.ones((3, 3)))
    branches = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        ang = np.arctan2(ys.mean() + y0 - cy, xs.mean() + x0 - cx)
        branches.append(ang)
    return branches


def angle_diff(a, b):
    d = abs(a - b) % (2 * np.pi)
    return min(d, 2 * np.pi - d)


def _measure_subpixel_orthogonal_width(vesselness, dist, skel_pt, branch_ang, art_angles, center, r_dist):
    """
    Nguyen vd. (2013, IEEE TBME) yöntemine göre damar merkez çizgisine DİK (ortogonal)
    bir kesit doğrusu atar ve damarın piksel-altı (sub-pixel) FWHM çapını ölçer.
    Kesişim yakınında atardamar doğrultusuna giren pikselleri maskeleyerek iki damarın
    birleşik lekesinin (X-blob) çapı şişirmesini engeller.
    """
    h, w = dist.shape
    py, px = skel_pt
    if not (2 <= py < h - 2 and 2 <= px < w - 2):
        return None

    base_dist_w = float(2.0 * dist[py, px])
    max_half = max(min(base_dist_w * 1.35 + 3.0, 18.0), 5.0)

    # Damar doğrultusuna dik birim normal vektör: (-sin(theta), cos(theta))
    ny = -np.sin(branch_ang)
    nx = np.cos(branch_ang)

    cy, cx = center
    steps = np.linspace(-max_half, max_half, 41)  # sub-pixel adımlar
    prof = np.zeros_like(steps, dtype=np.float32)
    valid_side = np.ones_like(steps, dtype=bool)

    for idx, s in enumerate(steps):
        sy = py + s * ny
        sx = px + s * nx
        iy, ix = int(round(sy)), int(round(sx))
        if not (0 <= iy < h and 0 <= ix < w):
            valid_side[idx] = False
            continue
        # Kesişime çok yakınken (r_dist < 18), dikey kesit atardamar koridoruna girerse o tarafı kes
        if r_dist < 18 and art_angles:
            ang_from_c = np.arctan2(sy - cy, sx - cx)
            if min(angle_diff(ang_from_c, aa) for aa in art_angles) < np.deg2rad(22):
                if abs(s) > 1.5:
                    valid_side[idx] = False
                    continue
        prof[idx] = vesselness[iy, ix]

    center_idx = len(steps) // 2
    peak_val = max(float(prof[center_idx]), float(prof[max(0, center_idx - 1):center_idx + 2].max()))
    if peak_val <= 0.05:
        return base_dist_w

    # Sağ ve sol yarıda Full-Width at Half-Maximum (FWHM) sub-pixel kenar bulma
    half_thr = 0.48 * peak_val

    # Pozitif yön (+s)
    r_pos = None
    for idx in range(center_idx, len(steps)):
        if not valid_side[idx] or prof[idx] < half_thr:
            if idx > center_idx and valid_side[idx]:
                # Doğrusal sub-pixel interpolasyon
                v_prev, v_cur = prof[idx - 1], prof[idx]
                frac = (v_prev - half_thr) / max(v_prev - v_cur, 1e-5)
                r_pos = steps[idx - 1] + frac * (steps[idx] - steps[idx - 1])
            else:
                r_pos = steps[max(center_idx, idx - 1)]
            break
    if r_pos is None:
        r_pos = base_dist_w * 0.5

    # Negatif yön (-s)
    r_neg = None
    for idx in range(center_idx, -1, -1):
        if not valid_side[idx] or prof[idx] < half_thr:
            if idx < center_idx and valid_side[idx]:
                v_prev, v_cur = prof[idx + 1], prof[idx]
                frac = (v_prev - half_thr) / max(v_prev - v_cur, 1e-5)
                r_neg = abs(steps[idx + 1] + frac * (steps[idx] - steps[idx + 1]))
            else:
                r_neg = abs(steps[min(center_idx, idx + 1)])
            break
    if r_neg is None:
        r_neg = base_dist_w * 0.5

    # Eğer bir taraf atardamar tarafından perdelendiyse diğer yarıçapın simetriğini kullan
    if not all(valid_side[center_idx:center_idx + 5]) and all(valid_side[center_idx - 4:center_idx + 1]):
        r_pos = r_neg
    elif not all(valid_side[center_idx - 4:center_idx + 1]) and all(valid_side[center_idx:center_idx + 5]):
        r_neg = r_pos

    fwhm_w = float(r_pos + r_neg)
    # Kesişime yakın bölgede ikili maske X-lekesi şiştiği için sub-pixel FWHM ağırlığını artır
    if r_dist < 16:
        blended = 0.70 * min(fwhm_w, base_dist_w * 1.05) + 0.30 * min(base_dist_w, fwhm_w * 1.15)
    else:
        blended = 0.50 * fwhm_w + 0.50 * base_dist_w
    return float(np.clip(blended, 1.2, 28.0))


def trace_branch_profile(skel, dist, vesselness, center, angle, r_start, r_end, art_angles=None, step=1):
    """
    Kol doğrultusunda iskeleti takip ederek her mesafedeki:
    1) İskelet koordinatlarını
    2) Sub-pixel ortogonal çap profilini W(d)
    3) Yerel açı yönelimini (Salus sapma açısı için) döndürür.
    """
    cy, cx = center
    h, w = skel.shape
    dists = []
    widths = []
    coords_y = []
    coords_x = []

    cur_ang = angle
    for r in np.arange(r_start, r_end, step):
        py = cy + r * np.sin(cur_ang)
        px = cx + r * np.cos(cur_ang)
        iy, ix = int(round(py)), int(round(px))
        best = None
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                yy, xx = iy + dy, ix + dx
                if 0 <= yy < h and 0 <= xx < w and skel[yy, xx]:
                    dd = dy * dy + dx * dx
                    if best is None or dd < best[0]:
                        best = (dd, yy, xx)
        if best is not None:
            by, bx = best[1], best[2]
            # Salus S-kıvrımını kaçırmamak için takip açısını yumuşakça güncelle
            obs_ang = np.arctan2(by - cy, bx - cx)
            if angle_diff(obs_ang, angle) < np.deg2rad(42):
                cur_ang = 0.75 * cur_ang + 0.25 * obs_ang
            w_val = _measure_subpixel_orthogonal_width(
                vesselness, dist, (by, bx), cur_ang, art_angles or (), center, r
            )
            if w_val is not None:
                dists.append(float(r))
                widths.append(w_val)
                coords_y.append(by)
                coords_x.append(bx)

    return {
        "distances": np.array(dists, dtype=np.float32),
        "widths": np.array(widths, dtype=np.float32),
        "coords": (np.array(coords_y, dtype=np.int32), np.array(coords_x, dtype=np.int32)),
    }


def analyze_crossings(rgb, enh, mask, skel, od=None, vesselness=None, channels=None):
    if vesselness is None:
        fov = (rgb[:, :, 0] > 15) | (rgb[:, :, 1] > 15)
        vesselness = compute_vesselness(enh, fov)
    if channels is None:
        channels = prepare_color_channels(rgb, enh)

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    med_w = float(2.0 * np.median(dist[skel])) if skel.any() else 4.0
    centers = find_junction_clusters(skel)
    results = []

    crossing_model = _load_model("crossing_validator.pkl")

    for c in centers:
        # Klinik olarak AV nicking optik disk DIŞINDA değerlendirilir
        if od is not None and np.hypot(c[0] - od[0][0], c[1] - od[0][1]) < 1.35 * od[1]:
            continue

        local_w = float(2.0 * dist[max(0, c[0] - 6):c[0] + 7, max(0, c[1] - 6):c[1] + 7].max())
        r_in = int(np.clip(0.85 * local_w + 4, 9, 3.0 * med_w + 8))
        r_out = r_in + 4

        angs = ring_branches(skel, c, r_in, r_out)
        outer = ring_branches(skel, c, r_in + 16, r_in + 22)

        # Eğer iç halkada 3 kol çıkıp dış halkada 4 kol varsa (kesişim lekesi geniş olduğu için),
        # biraz daha geniş bir iç halkadan (r_in+4 .. r_in+8) 4 kolu yakala
        if len(angs) != 4 and len(outer) == 4:
            angs_alt = ring_branches(skel, c, r_in + 4, r_in + 8)
            if len(angs_alt) == 4:
                angs = angs_alt
                r_in = r_in + 3

        if len(angs) != 4:
            continue

        # Kolların en az 3'ü dış halkada da devam etmeli (gürültü/sahte kolları eler)
        if sum(any(angle_diff(a, o) < np.deg2rad(32) for o in outer) for a in angs) < 3:
            continue

        # Karşılıklı kolları eşle (açı farkı ~180°)
        best = None
        for pair in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
            s = sum(abs(np.pi - angle_diff(angs[a], angs[b])) for a, b in pair)
            if best is None or s < best[0]:
                best = (s, pair)

        # Salus Belirtisi: Kesişimde S-şeklinde bükülen toplardamarları yakalamak için
        # dış halkada devam eden damarlarda 52°'ye kadar açısal sapmaya izin veriyoruz.
        pair_deviations = [abs(np.pi - angle_diff(angs[a], angs[b])) for a, b in best[1]]
        if max(pair_deviations) > np.deg2rad(52):
            continue

        # İki damar kolu çifti için ön izleme ve A/V (Atardamar/Toplardamar) olasılığı hesapla
        prelim_Branches = []
        valid_pairs = True
        for pair_idx, (a, b) in enumerate(best[1]):
            other_pair = best[1][1 - pair_idx]
            other_angs = (angs[other_pair[0]], angs[other_pair[1]])

            p1 = trace_branch_profile(skel, dist, vesselness, c, angs[a], r_in, r_in + 42, art_angles=other_angs)
            p2 = trace_branch_profile(skel, dist, vesselness, c, angs[b], r_in, r_in + 42, art_angles=other_angs)

            if len(p1["widths"]) < 6 or len(p2["widths"]) < 6:
                valid_pairs = False
                break

            all_ys = np.concatenate([p1["coords"][0], p2["coords"][0]])
            all_xs = np.concatenate([p1["coords"][1], p2["coords"][1]])
            av_feats = extract_branch_av_features((all_ys, all_xs), dist, channels, med_w)
            p_artery = predict_artery_probability(av_feats)

            # Yakın (nicking bölgesi: r_in .. r_in+10) ve uzak (referans: r_in+18 .. r_in+42) çaplar
            near1 = p1["widths"][p1["distances"] <= r_in + 10]
            near2 = p2["widths"][p2["distances"] <= r_in + 10]
            far1 = p1["widths"][p1["distances"] >= r_in + 16]
            far2 = p2["widths"][p2["distances"] >= r_in + 16]

            if len(near1) < 2 or len(near2) < 2 or len(far1) < 3 or len(far2) < 3:
                valid_pairs = False
                break

            w_near1 = float(np.percentile(near1, 30))
            w_near2 = float(np.percentile(near2, 30))
            w_far1 = float(np.median(far1))
            w_far2 = float(np.median(far2))

            # Uzak halkadaki gerçek doğrultu açısı ile iç halkadaki açı arasındaki fark (Salus sapması)
            far_ang1 = np.arctan2(p1["coords"][0][-1] - c[0], p1["coords"][1][-1] - c[1])
            far_ang2 = np.arctan2(p2["coords"][0][-1] - c[0], p2["coords"][1][-1] - c[1])
            near_dev_deg = float(np.rad2deg(abs(np.pi - angle_diff(angs[a], angs[b]))))
            far_dev_deg = float(np.rad2deg(abs(np.pi - angle_diff(far_ang1, far_ang2))))
            salus_deg = float(max(near_dev_deg, abs(near_dev_deg - far_dev_deg)))

            prelim_Branches.append({
                "angles": (angs[a], angs[b]),
                "p1": p1,
                "p2": p2,
                "w_near1": w_near1,
                "w_near2": w_near2,
                "w_far1": w_far1,
                "w_far2": w_far2,
                "near": float(0.5 * (w_near1 + w_near2)),
                "far": float(0.5 * (w_far1 + w_far2)),
                "p_artery": p_artery,
                "av_feats": av_feats,
                "salus_deg": salus_deg,
            })

        if not valid_pairs or len(prelim_Branches) != 2:
            continue

        b0, b1 = prelim_Branches[0], prelim_Branches[1]
        # Çok ince kılcal gürültüleri ele
        if min(b0["far"], b1["far"]) < 0.55 * med_w:
            continue

        # Model 2 (Crossing Validator) için kesişim doğrulama öznitelikleri
        crossing_angle_deg = float(np.rad2deg(min(
            angle_diff(b0["angles"][0], b1["angles"][0]),
            angle_diff(b0["angles"][0], b1["angles"][1]),
        )))
        av_prob_diff = abs(b0["p_artery"] - b1["p_artery"])
        width_ratio = max(b0["far"], b1["far"]) / max(min(b0["far"], b1["far"]), 1e-5)
        c_feats = np.array([
            av_prob_diff,
            max(b0["p_artery"], b1["p_artery"]),
            min(b0["p_artery"], b1["p_artery"]),
            width_ratio,
            crossing_angle_deg / 90.0,
            max(b0["salus_deg"], b1["salus_deg"]) / 45.0,
            abs(b0["av_feats"][1] - b1["av_feats"][1]),   # yeşil kanal kontrast farkı
            abs(b0["av_feats"][5] - b1["av_feats"][5]),   # R-G farkı
            abs(b0["av_feats"][11] - b1["av_feats"][11]), # Central Light Reflex farkı
        ], dtype=np.float32)

        crossing_conf = 0.75
        if crossing_model is not None:
            try:
                crossing_conf = float(crossing_model["model"].predict_proba(c_feats.reshape(1, -1))[0, 1])
                if crossing_conf < 0.28:
                    continue
            except Exception:
                pass

        # Hangi çift Toplardamar (Vein), hangi çift Atardamar (Artery)?
        # Model 1 olasılığı (1 - p_artery) ile kalınlık oranını hibrit olarak birleştir
        vein_score_0 = (1.0 - b0["p_artery"]) + 0.25 * (b0["far"] / max(b0["far"] + b1["far"], 1e-5))
        vein_score_1 = (1.0 - b1["p_artery"]) + 0.25 * (b1["far"] / max(b0["far"] + b1["far"], 1e-5))
        vi = 0 if vein_score_0 >= vein_score_1 else 1
        vein, art = prelim_Branches[vi], prelim_Branches[1 - vi]

        # 1) Gunn Belirtisi (AV Nicking Daralma Oranı)
        mean_narrowing = max(0.0, 1.0 - vein["near"] / max(vein["far"], 1e-6))
        side1_narrowing = max(0.0, 1.0 - vein["w_near1"] / max(vein["w_far1"], 1e-6))
        side2_narrowing = max(0.0, 1.0 - vein["w_near2"] / max(vein["w_far2"], 1e-6))
        # Nguyen vd. (2013): iki taraftaki daralma ile ortalama daralmanın ağırlıklı birleşimi
        effective_narrowing = float(0.65 * mean_narrowing + 0.35 * max(side1_narrowing, side2_narrowing))

        # 2) Venous Banking (Toplardamarın bir tarafta şişmesi / kan birikimi asimetrisi)
        banking_index = float(abs(vein["w_far1"] - vein["w_far2"]) / max(vein["w_far1"], vein["w_far2"], 1e-5))

        # 3) Salus Belirtisi (Toplardamarın kesişimde S şeklinde açısal sapması)
        salus_angle_deg = float(vein["salus_deg"])

        # Çap değişim profili eğrisi W(d) (Streamlit grafiği için)
        d_grid = np.arange(0, 36, 1, dtype=np.float32)
        p1_d = vein["p1"]["distances"] - r_in
        p2_d = vein["p2"]["distances"] - r_in
        w1_interp = np.interp(d_grid, p1_d, vein["p1"]["widths"]) if len(p1_d) > 1 else np.full_like(d_grid, vein["far"])
        w2_interp = np.interp(d_grid, p2_d, vein["p2"]["widths"]) if len(p2_d) > 1 else np.full_like(d_grid, vein["far"])
        w_mean_curve = 0.5 * (w1_interp + w2_interp)

        results.append({
            "center": c,
            "vein_near": float(vein["near"]),
            "vein_far": float(vein["far"]),
            "vein_near_sides": (float(vein["w_near1"]), float(vein["w_near2"])),
            "vein_far_sides": (float(vein["w_far1"]), float(vein["w_far2"])),
            "art_far": float(art["far"]),
            "narrowing": effective_narrowing,
            "raw_narrowing": float(mean_narrowing),
            "max_side_narrowing": float(max(side1_narrowing, side2_narrowing)),
            "banking_index": banking_index,
            "salus_angle_deg": salus_angle_deg,
            "crossing_angle_deg": crossing_angle_deg,
            "artery_prob": float(art["p_artery"]),
            "vein_prob": float(1.0 - vein["p_artery"]),
            "crossing_confidence": crossing_conf,
            "r_in": r_in,
            "vein_angles": vein["angles"],
            "art_angles": art["angles"],
            "profile_d": d_grid.tolist(),
            "profile_w_mean": w_mean_curve.tolist(),
            "profile_w_side1": w1_interp.tolist(),
            "profile_w_side2": w2_interp.tolist(),
        })

    # Sonuçları daralma şiddetine göre (en yüksekten düşüğe) sırala
    results.sort(key=lambda r: r["narrowing"], reverse=True)
    return results, dist


# ================================================================
# 7) ERKEN TEŞHİS ÇOKLU BİYOBELİRTEÇ VE MODEL 3 (HASTA RİSK SKORU)
# ================================================================
def classify(ratio):
    if ratio >= 0.30:
        return "YÜKSEK", (255, 40, 40)
    if ratio >= 0.15:
        return "ORTA", (255, 170, 0)
    return "NORMAL", (60, 220, 60)


def compute_early_diagnosis(results, av_info, tortuosity_mean, tortuosity_p80):
    """
    4 Klinik Biyobelirteç Ailesini Birleştirerek (Gunn + Salus + AVR + Kıvrımlılık)
    Eğitilmiş Model 3 (models/early_diagnosis_model.pkl) ile Erken Teşhis Risk Skoru üretir.
    """
    if results:
        narrowings = [r["narrowing"] for r in results]
        salus_angles = [r["salus_angle_deg"] for r in results]
        bankings = [r["banking_index"] for r in results]
        max_narrowing = float(max(narrowings))
        top3_narrowing = float(np.mean(sorted(narrowings, reverse=True)[:3]))
        high_count = int(sum(n >= 0.25 for n in narrowings))
        mod_count = int(sum(0.15 <= n < 0.25 for n in narrowings))
        max_salus = float(max(salus_angles))
        mean_salus = float(np.mean(sorted(salus_angles, reverse=True)[:3]))
        mean_banking = float(np.mean(bankings))
    else:
        max_narrowing = 0.0
        top3_narrowing = 0.0
        high_count = 0
        mod_count = 0
        max_salus = 0.0
        mean_salus = 0.0
        mean_banking = 0.0

    avr = float(av_info.get("avr", 0.72))

    # 8 boyutlu hasta biyobelirteç vektörü (Model 3 girdisi)
    patient_feats = np.array([
        max_narrowing,
        top3_narrowing,
        float(high_count),
        max_salus / 45.0,
        mean_salus / 45.0,
        mean_banking,
        avr,
        tortuosity_mean,
    ], dtype=np.float32)

    # Klinik bileşen skorları (0..100 arası, açıklanabilirlik için)
    gunn_score = float(np.clip((0.6 * max_narrowing + 0.4 * top3_narrowing) / 0.38 * 100.0, 0.0, 100.0))
    salus_score = float(np.clip((0.6 * max_salus + 0.4 * mean_salus) / 32.0 * 100.0, 0.0, 100.0))
    # Normal AVR >= 0.70; 0.55 ve altı yüksek risk
    avr_score = float(np.clip((0.74 - avr) / 0.20 * 100.0, 0.0, 100.0))
    # Normal kıvrımlılık ~1.04; 1.12 ve üzeri yüksek kıvrımlılık
    tort_score = float(np.clip((tortuosity_mean - 1.04) / 0.08 * 100.0, 0.0, 100.0))

    heuristic_prob = float(np.clip(
        (0.42 * gunn_score + 0.20 * salus_score + 0.23 * avr_score + 0.15 * tort_score) / 100.0,
        0.02, 0.98
    ))

    diag_model = _load_model("early_diagnosis_model.pkl")
    if diag_model is not None:
        try:
            ml_prob = float(diag_model["model"].predict_proba(patient_feats.reshape(1, -1))[0, 1])
            # ML olasılığı ile klinik biyobelirteç skorunun dengeli birleşimi
            risk_prob = float(0.65 * ml_prob + 0.35 * heuristic_prob)
        except Exception:
            risk_prob = heuristic_prob
    else:
        risk_prob = heuristic_prob

    risk_pct = round(risk_prob * 100.0, 1)
    if risk_pct >= 60.0 or (max_narrowing >= 0.32 and high_count >= 2):
        risk_level = "YÜKSEK RİSK (Erken/Orta Evre Hipertansif Retinopati Bulgusu)"
        badge_color = "red"
    elif risk_pct >= 35.0 or max_narrowing >= 0.20:
        risk_level = "ORTA RİSK (Erken Evre Damarsal Değişim Şüphesi - Takip Önerilir)"
        badge_color = "orange"
    else:
        risk_level = "DÜŞÜK RİSK (Normal Retinal Mikrovasküler Yapı)"
        badge_color = "green"

    findings = []
    if max_narrowing >= 0.30:
        findings.append(f"Gunn Belirtisi (AV Nicking): Belirgin toplardamar sıkışması tespit edildi (Maks. daralma %{100*max_narrowing:.0f}, {high_count} yüksek riskli kesişim).")
    elif max_narrowing >= 0.15:
        findings.append(f"Gunn Belirtisi (AV Nicking): Hafif/Orta düzeyde toplardamar daralması mevcut (Maks. %{100*max_narrowing:.0f}).")
    else:
        findings.append("Gunn Belirtisi (AV Nicking): Çaprazlaşma noktalarında patolojik daralma saptanmadı.")

    if max_salus >= 18.0:
        findings.append(f"Salus Belirtisi (Sapma Açısı): Kesişimde S-şekilli toplardamar sapması gözlemlendi (Maks. {max_salus:.1f}°).")
    if avr < 0.65:
        findings.append(f"Atardamar/Toplardamar Çap Oranı (AVR = {avr:.2f}): Normal referansın (<0.67) altında, erken atardamar daralması bulgusu.")
    else:
        findings.append(f"Atardamar/Toplardamar Çap Oranı (AVR = {avr:.2f}): Normal fizyolojik aralıkta.")

    if tortuosity_mean >= 1.08:
        findings.append(f"Damar Kıvrımlılığı (İndeks = {tortuosity_mean:.3f}): Artmış vasküler kıvrımlılık (hipertansif stres göstergesi).")

    return {
        "risk_probability": risk_prob,
        "risk_Percentage": risk_pct,
        "risk_level": risk_level,
        "badge_color": badge_color,
        "features": patient_feats.tolist(),
        "biomarkers": {
            "max_narrowing": max_narrowing,
            "top3_narrowing": top3_narrowing,
            "high_nicking_count": high_count,
            "moderate_nicking_count": mod_count,
            "max_salus_deg": max_salus,
            "mean_salus_deg": mean_salus,
            "mean_banking": mean_banking,
            "avr": avr,
            "crae_px": av_info.get("crae_px", 0.0),
            "crve_px": av_info.get("crve_px", 0.0),
            "tortuosity_mean": tortuosity_mean,
            "tortuosity_p80": tortuosity_p80,
        },
        "component_scores": {
            "Gunn (AV Nicking Daralması)": round(gunn_score, 1),
            "Salus (Kesişim Sapma Açısı)": round(salus_score, 1),
            "AVR (Atardamar İncelmesi)": round(avr_score, 1),
            "Tortuosity (Damar Kıvrımlılığı)": round(tort_score, 1),
        },
        "clinical_findings": findings,
    }


def draw_results(rgb, results):
    out = rgb.copy()
    for i, r in enumerate(results, 1):
        cy, cx = r["center"]
        _, col = classify(r["narrowing"])
        cv2.circle(out, (cx, cy), r["r_in"] + 7, col, 3)
        cv2.putText(out, str(i), (cx + r["r_in"] + 6, cy - r["r_in"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, str(i), (cx + r["r_in"] + 6, cy - r["r_in"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    return out


def run_pipeline(path):
    rgb, enh, fov = load_and_preprocess(path)
    mask, vesselness = segment_vessels(enh, fov, return_vesselness=True)
    skel = skeletonize(mask)
    skel = prune_spurs(skel)
    od = detect_optic_disc(rgb, fov, mask=mask)
    channels = prepare_color_channels(rgb, enh)

    results, dist = analyze_crossings(rgb, enh, mask, skel, od, vesselness=vesselness, channels=channels)
    med_w = float(2.0 * np.median(dist[skel])) if skel.any() else 4.0
    av_info = build_av_segmentation_map(rgb, mask, skel, dist, channels, med_w)
    tort_mean, tort_p80, tort_segs = compute_vessel_tortuosity(skel)
    diagnosis = compute_early_diagnosis(results, av_info, tort_mean, tort_p80)

    ann = draw_results(rgb, results)
    (oy, ox), orad = od
    cv2.circle(ann, (ox, oy), int(1.35 * orad), (120, 200, 255), 2)

    return dict(
        rgb=rgb,
        enh=enh,
        fov=fov,
        mask=mask,
        vesselness=vesselness,
        skel=skel,
        od=od,
        dist=dist,
        av_info=av_info,
        tortuosity=dict(mean=tort_mean, p80=tort_p80, segments=len(tort_segs)),
        diagnosis=diagnosis,
        results=results,
        annotated=ann,
    )


def text_report(name, res):
    rs = res["results"]
    diag = res.get("diagnosis", {})
    bm = diag.get("biomarkers", {})
    lines = [
        f"==========================================================================",
        f"  RETİNAL AV NICKING VE HİPERTANSİF RETİNOPATİ ERKEN TEŞHİS RAPORU: {name}",
        f"==========================================================================",
        f"Erken Teşhis Risk Skoru : %{diag.get('risk_Percentage', 0):.1f}  -->  {diag.get('risk_level', '-')}",
        f"Bulunan A/V Çaprazlaşma : {len(rs)} adet",
        f"AVR (Atar/Toplar Oranı) : {bm.get('avr', 0):.2f}  (CRAE: {bm.get('crae_px', 0):.1f}px / CRVE: {bm.get('crve_px', 0):.1f}px)",
        f"Damar Kıvrımlılığı      : {bm.get('tortuosity_mean', 1.0):.3f} (80. Persentil: {bm.get('tortuosity_p80', 1.0):.3f})",
        f"--------------------------------------------------------------------------",
        f"{'No':>3} {'Konum(x,y)':>12} {'Toplar Uzak':>11} {'Toplar Yakın':>12} {'Daralma':>8} {'Salus Açısı':>11} {'A/V Güven':>10}  Seviye",
    ]
    for i, r in enumerate(rs, 1):
        lvl, _ = classify(r["narrowing"])
        av_conf = 0.5 * (r.get("artery_prob", 0.5) + r.get("vein_prob", 0.5))
        lines.append(
            f"{i:>3} {str((r['center'][1], r['center'][0])):>12} "
            f"{r['vein_far']:>9.1f}px {r['vein_near']:>10.1f}px "
            f"{100*r['narrowing']:>7.0f}% "
            f"{r.get('salus_angle_deg', 0.0):>10.1f}° "
            f"{100*av_conf:>9.0f}%  {lvl}"
        )
    lines.append("--------------------------------------------------------------------------")
    lines.append("Klinik Bulgular:")
    for f in diag.get("clinical_findings", []):
        lines.append(f"  * {f}")
    return "\n".join(lines)
