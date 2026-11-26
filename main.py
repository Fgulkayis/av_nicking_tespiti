import os
import glob

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # grafikleri ekranda açmadan doğrudan dosyaya kaydet
import matplotlib.pyplot as plt

from src.av_nicking import run_pipeline, text_report, classify

PROJE_KOKU = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJE_KOKU, "data")
RESULTS_DIR = os.path.join(PROJE_KOKU, "results")
UZANTILAR = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def save_full(name, res, out_path):
    """4 adımı yan yana gösteren genel görseli kaydeder."""
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(res["rgb"]); ax[0].set_title("1) Orijinal")
    ax[1].imshow(res["enh"], cmap="gray"); ax[1].set_title("2) Yeşil kanal + CLAHE")
    v = np.dstack([res["mask"] * 90] * 3).astype(np.uint8); v[res["skel"]] = [255, 255, 0]
    ax[2].imshow(v); ax[2].set_title("3) Damar maskesi + iskelet")
    ax[3].imshow(res["annotated"]); ax[3].set_title("4) Çaprazlaşmalar (yeşil/turuncu/kırmızı)\nmavi çember: optik disk (hariç)")
    for a in ax:
        a.axis("off")
    plt.suptitle(name, fontsize=14); plt.tight_layout()
    fig.savefig(out_path, dpi=100); plt.close(fig)


def save_crossings(res, out_path, R=70):
    """Her çaprazlaşmanın yakın görünümünü kaydeder. Çaprazlaşma yoksa False döner."""
    rs = res["results"]
    if not rs:
        return False
    fig, ax = plt.subplots(len(rs), 2, figsize=(8, 4 * len(rs)), squeeze=False)
    pad = lambda a: np.pad(a, ((R, R), (R, R)) + ((0, 0),) * (a.ndim - 2))
    for i, r in enumerate(rs):
        cy, cx = r["center"]
        crop = pad(res["rgb"])[cy:cy + 2*R, cx:cx + 2*R]
        m = (pad(res["mask"])[cy:cy + 2*R, cx:cx + 2*R] * 255).astype(np.uint8)
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
        m[pad(res["skel"])[cy:cy + 2*R, cx:cx + 2*R]] = [255, 0, 0]
        for ang in r["vein_angles"]:
            cv2.line(m, (R, R), (int(R + 60*np.cos(ang)), int(R + 60*np.sin(ang))), (0, 120, 255), 2)
        lvl, _ = classify(r["narrowing"])
        ax[i, 0].imshow(crop); ax[i, 0].set_title(f"#{i+1}  daralma %{100*r['narrowing']:.0f} ({lvl})")
        ax[i, 1].imshow(m); ax[i, 1].set_title(f"mavi = toplardamar  uzak {r['vein_far']:.1f}px → yakın {r['vein_near']:.1f}px")
        ax[i, 0].axis("off"); ax[i, 1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100); plt.close(fig)
    return True


def main():
    # data/ ve alt klasörlerindeki tüm görüntüleri bul
    goruntuler = sorted(
        p for p in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
        if p.lower().endswith(UZANTILAR)
    )
    print(f"{len(goruntuler)} görüntü analiz edilecek")
    if not goruntuler:
        print(f"'{DATA_DIR}' klasörüne en az bir fundus görüntüsü koyup tekrar çalıştırın.")
        return

    for g in goruntuler:
        ad = os.path.splitext(os.path.basename(g))[0]
        hedef = os.path.join(RESULTS_DIR, ad)
        os.makedirs(hedef, exist_ok=True)

        res = run_pipeline(g)
        rapor = text_report(os.path.basename(g), res)
        print(rapor + "\n")

        with open(os.path.join(hedef, "rapor.txt"), "w", encoding="utf-8") as f:
            f.write(rapor + "\n")
        save_full(ad, res, os.path.join(hedef, "genel_gorunum.png"))
        save_crossings(res, os.path.join(hedef, "caprazlasmalar.png"))

    print(f"Sonuçlar kaydedildi: {RESULTS_DIR}")


if __name__ == "__main__":
    main()