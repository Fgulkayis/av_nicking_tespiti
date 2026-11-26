import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, disk


def _rm_small_objects(m, n):
    # scikit-image sürüm farkı (0.26+ max_size, eskiler min_size)
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
    # cv2.imread Windows'ta Türkçe karakterli yolları okuyamaz (None döner).
    # Dosyayı önce bayt olarak okuyup sonra çözmek her yolda çalışır.
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


# ---------------------------------------------------------------- 1) ÖN İŞLEME
def load_and_preprocess(path, target_width=1000):
    bgr = _imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    scale = target_width / w
    bgr = cv2.resize(bgr, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # FOV maskesi: kırmızı kanal karanlık arka plandan ayrılır
    red = rgb[:, :, 0]
    fov = (red > 20).astype(np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fov = cv2.erode(fov, np.ones((15, 15), np.uint8))  # kenar parlamasını at
    green = rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enh = clahe.apply(green)
    enh = cv2.medianBlur(enh, 3)
    return rgb, enh, fov.astype(bool)


# ---------------------------------------------------------- 2) SEGMENTASYON
def segment_vessels(enh, fov, min_size=400):
    # Damarlar arka plandan koyu -> black-top-hat damarları parlak yapar
    th = np.zeros(enh.shape, np.float32)
    for r in (7, 11, 15):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        th = np.maximum(th, cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, k).astype(np.float32))
    th[~fov] = 0
    th = cv2.GaussianBlur(th, (3, 3), 0)
    vals = th[fov]
    t = np.percentile(vals, 85)  # en güçlü %15 piksel damar adayı
    mask = (th > t) & fov
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    mask = _rm_small_objects(mask, min_size)
    # küçük delikleri kapat
    mask = ndi.binary_closing(mask, structure=disk(1))
    # Damar ortasındaki parlak yansıma (central reflex) maskede delik bırakır;
    # bu delikler iskelette sahte "merdiven" kesişimleri üretir -> doldur
    mask = _rm_small_holes(mask, 300)
    return mask


def detect_optic_disc(rgb, fov):
    """Optik disk = görüntünün en parlak, yuvarlak bölgesi (kırmızı+yeşil)."""
    w = rgb.shape[1]
    lum = (0.5 * rgb[:, :, 0] + 0.5 * rgb[:, :, 1]).astype(np.float32)
    lum[~fov] = 0
    k = int(w * 0.08) | 1
    blur = cv2.GaussianBlur(lum, (k, k), 0)
    _, _, _, (x, y) = cv2.minMaxLoc(blur)
    radius = int(w * 0.075)   # tipik optik disk yarıçapı (görüntü genişliğine göre)
    return (y, x), radius


# ---------------------------------------------------------- 3) İSKELET
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
        # uç noktası olan ve kısa olan dallar = spur
        dil_j = ndi.binary_dilation(junction, structure=np.ones((5, 5)))
        touches_j = ndi.maximum(dil_j, lab, index=np.arange(1, n + 1))
        remove = np.where((sizes < length) & (has_end > 0) & (touches_j > 0))[0] + 1
        if len(remove) == 0:
            break
        skel[np.isin(lab, remove)] = False
        skel = skeletonize(skel)
    return skel


# ---------------------------------------------------------- 4) KESİŞİMLER
def find_junction_clusters(skel, merge_radius=8):
    nb = neighbor_count(skel)
    jpts = nb >= 3
    jd = ndi.binary_dilation(jpts, structure=disk(merge_radius))
    lab, n = ndi.label(jd)
    centers = ndi.center_of_mass(jpts, lab, index=np.arange(1, n + 1))
    return [(int(round(r)), int(round(c))) for r, c in centers if not np.isnan(r)]


def ring_branches(skel, center, r_in, r_out):
    """Merkez etrafındaki halkada iskeleti kesen kolları bul."""
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


def trace_branch_widths(skel, dist, center, angle, r_start, r_end, step=1):
    """Kol doğrultusunda iskeleti takip edip her mesafedeki çapı ölç."""
    cy, cx = center
    h, w = skel.shape
    widths = []
    for r in np.arange(r_start, r_end, step):
        # beklenen nokta etrafında (±3px) en yakın iskelet pikselini ara
        py = cy + r * np.sin(angle)
        px = cx + r * np.cos(angle)
        iy, ix = int(round(py)), int(round(px))
        best = None
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                yy, xx = iy + dy, ix + dx
                if 0 <= yy < h and 0 <= xx < w and skel[yy, xx]:
                    dd = dy * dy + dx * dx
                    if best is None or dd < best[0]:
                        best = (dd, yy, xx)
        if best is not None:
            widths.append(2.0 * dist[best[1], best[2]])
    return widths


def angle_diff(a, b):
    d = abs(a - b) % (2 * np.pi)
    return min(d, 2 * np.pi - d)


# ---------------------------------------------------------- 5) ANALİZ
def analyze_crossings(rgb, enh, mask, skel, od=None):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    med_w = 2 * np.median(dist[skel]) if skel.any() else 4
    centers = find_junction_clusters(skel)
    green = rgb[:, :, 1].astype(np.float32)
    results = []
    for c in centers:
        # Klinik olarak AV nicking optik disk DIŞINDA değerlendirilir
        if od is not None and np.hypot(c[0] - od[0][0], c[1] - od[0][1]) < 1.4 * od[1]:
            continue
        local_w = 2 * dist[max(0, c[0]-6):c[0]+7, max(0, c[1]-6):c[1]+7].max()
        # halka yarıçapı: kesişimde birleşen blob'un hemen dışı
        r_in = int(np.clip(0.9 * local_w + 4, 10, 3 * med_w + 8))
        r_out = r_in + 4
        angs = ring_branches(skel, c, r_in, r_out)
        if len(angs) != 4:
            continue  # 3 kol = dallanma, diğerleri belirsiz
        # Kollar gerçek damar mı? Daha geniş halkada da aynı yönlerde görünmeli
        outer = ring_branches(skel, c, r_in + 20, r_in + 24)
        if sum(any(angle_diff(a, o) < np.deg2rad(25) for o in outer) for a in angs) < 3:
            continue
        # karşılıklı kolları eşle (açı farkı ~180°)
        best = None
        for pair in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
            s = sum(abs(np.pi - angle_diff(angs[a], angs[b])) for a, b in pair)
            if best is None or s < best[0]:
                best = (s, pair)
        # Çaprazlaşmada iki damar da kesişimden DÜZ geçer (her çift ~180°).
        # Dallanmada ise kollar açılı ayrılır -> ele
        if max(abs(np.pi - angle_diff(angs[a], angs[b])) for a, b in best[1]) > np.deg2rad(30):
            continue
        vessels = []
        for a, b in best[1]:
            info = {"angles": (angs[a], angs[b])}
            near, far, ints = [], [], []
            for ang in (angs[a], angs[b]):
                near += trace_branch_widths(skel, dist, c, ang, r_in, r_in + 8)
                far += trace_branch_widths(skel, dist, c, ang, r_in + 22, r_in + 45)
                # renk (koyuluk) örneği
                for r in range(r_in + 5, r_in + 30, 3):
                    y = int(round(c[0] + r * np.sin(ang)))
                    x = int(round(c[1] + r * np.cos(ang)))
                    if 0 <= y < green.shape[0] and 0 <= x < green.shape[1] and mask[y, x]:
                        ints.append(green[y, x])
            if len(near) < 4 or len(far) < 6:
                info = None
            else:
                info.update(near=float(np.median(near)), far=float(np.median(far)),
                            intensity=float(np.mean(ints)) if ints else 255.0)
            vessels.append(info)
        if None in vessels:
            continue
        # Sadece belirgin (ana) damarların çaprazlaşmasını değerlendir
        if min(v["far"] for v in vessels) < 0.6 * med_w:
            continue
        # Toplardamar: belirgin şekilde daha kalınsa kalın olan, değilse daha koyu olan
        f0, f1 = vessels[0]["far"], vessels[1]["far"]
        if max(f0, f1) / max(min(f0, f1), 1e-6) > 1.25:
            vi = 0 if f0 > f1 else 1
        else:
            vi = 0 if vessels[0]["intensity"] < vessels[1]["intensity"] else 1
        vein, art = vessels[vi], vessels[1 - vi]
        ratio = 1.0 - vein["near"] / max(vein["far"], 1e-6)
        results.append({
            "center": c, "vein_near": vein["near"], "vein_far": vein["far"],
            "art_far": art["far"], "narrowing": max(0.0, ratio), "r_in": r_in,
            "vein_angles": vein["angles"],
        })
    return results, dist


def classify(ratio):
    if ratio >= 0.30:
        return "YÜKSEK", (255, 40, 40)
    if ratio >= 0.15:
        return "ORTA", (255, 170, 0)
    return "NORMAL", (60, 220, 60)


def draw_results(rgb, results):
    out = rgb.copy()
    for i, r in enumerate(results, 1):
        cy, cx = r["center"]
        _, col = classify(r["narrowing"])
        cv2.circle(out, (cx, cy), r["r_in"] + 6, col, 3)
        cv2.putText(out, str(i), (cx + r["r_in"] + 6, cy - r["r_in"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, str(i), (cx + r["r_in"] + 6, cy - r["r_in"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    return out


def run_pipeline(path):
    rgb, enh, fov = load_and_preprocess(path)
    mask = segment_vessels(enh, fov)
    skel = skeletonize(mask)
    skel = prune_spurs(skel)
    od = detect_optic_disc(rgb, fov)
    results, dist = analyze_crossings(rgb, enh, mask, skel, od)
    ann = draw_results(rgb, results)
    (oy, ox), orad = od
    cv2.circle(ann, (ox, oy), int(1.4 * orad), (120, 200, 255), 2)
    return dict(rgb=rgb, enh=enh, fov=fov, mask=mask, skel=skel, od=od,
                results=results, annotated=ann)


def text_report(name, res):
    rs = res["results"]
    lines = [f"=== {name} ===", f"Bulunan A/V çaprazlaşma adayı: {len(rs)}"]
    lines.append(f"{'No':>3} {'Konum(x,y)':>12} {'Toplar uzak':>11} {'Toplar yakın':>12} {'Daralma':>8}  Seviye")
    for i, r in enumerate(rs, 1):
        lvl, _ = classify(r["narrowing"])
        lines.append(f"{i:>3} {str((r['center'][1], r['center'][0])):>12} "
                     f"{r['vein_far']:>9.1f}px {r['vein_near']:>10.1f}px "
                     f"{100*r['narrowing']:>7.0f}%  {lvl}")
    if rs:
        mx = max(rs, key=lambda r: r["narrowing"])
        lvl, _ = classify(mx["narrowing"])
        n_hi = sum(r["narrowing"] >= 0.30 for r in rs)
        lines.append(f"En belirgin daralma: %{100*mx['narrowing']:.0f} ({lvl}); "
                     f"yüksek daralmalı çaprazlaşma sayısı: {n_hi}")
    return "\n".join(lines)