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
MASKE_KLASORLERI = {"arteries", "veins", "masks", "arteries-and-veins", "vessel-segmentations", "__macosx"}


def find_fundus_images(data_dir=DATA_DIR):
    """data/ altındaki gerçek renkli fundus görüntülerini bulur (etiket/maske klasörlerini atlar)."""
    bulunan = []
    for root, dirs, files in os.walk(data_dir):
        # Maske klasörlerini atla
        dirs[:] = [d for d in dirs if d.lower() not in MASKE_KLASORLERI]
        for f in files:
            if f.lower().endswith(UZANTILAR):
                bulunan.append(os.path.join(root, f))
    return sorted(bulunan)


def save_full(name, res, out_path):
    """5 adımı yan yana gösteren genel erken teşhis görselini kaydeder."""
    fig, ax = plt.subplots(1, 5, figsize=(26, 5.8))
    ax[0].imshow(res["rgb"])
    ax[0].set_title("1) Orijinal Fundus")

    ax[1].imshow(res["enh"], cmap="gray")
    ax[1].set_title("2) Yeşil Kanal + CLAHE")

    v = np.dstack([res["mask"] * 90] * 3).astype(np.uint8)
    v[res["skel"]] = [255, 255, 0]
    ax[2].imshow(v)
    ax[2].set_title(f"3) Hibrit Damar Maskesi + İskelet\nKıvrımlılık: {res['tortuosity']['mean']:.3f}")

    av_vis = res["av_info"]["av_vis"]
    avr_val = res["av_info"]["avr"]
    ax[3].imshow(av_vis)
    ax[3].set_title(f"4) A/V Ayrımı (Kırmızı:Atar, Mavi:Toplar)\nAVR Çap Oranı: {avr_val:.2f}")

    ax[4].imshow(res["annotated"])
    diag = res["diagnosis"]
    ax[4].set_title(
        f"5) A/V Çaprazlaşmaları ({len(res['results'])} adet)\n"
        f"Erken Teşhis Riski: %{diag['risk_Percentage']:.1f}"
    )

    for a in ax:
        a.axis("off")

    plt.suptitle(f"{name}  |  {diag['risk_level']}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_crossings(res, out_path, R=70):
    """Her çaprazlaşmanın yakın görünümünü, yönlerini ve Sub-Pixel Çap Değişim Grafiğini kaydeder."""
    rs = res["results"]
    if not rs:
        return False
    fig, ax = plt.subplots(len(rs), 3, figsize=(13.5, 3.8 * len(rs)), squeeze=False)
    pad = lambda a: np.pad(a, ((R, R), (R, R)) + ((0, 0),) * (a.ndim - 2))

    for i, r in enumerate(rs):
        cy, cx = r["center"]
        crop = pad(res["rgb"])[cy:cy + 2 * R, cx:cx + 2 * R]
        m = (pad(res["mask"])[cy:cy + 2 * R, cx:cx + 2 * R] * 255).astype(np.uint8)
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
        m[pad(res["skel"])[cy:cy + 2 * R, cx:cx + 2 * R]] = [255, 220, 0]

        # Atardamar yönü (Kırmızı çizgi) ve Toplardamar yönü (Mavi çizgi)
        for ang in r.get("art_angles", ()):
            cv2.line(m, (R, R), (int(R + 58 * np.cos(ang)), int(R + 58 * np.sin(ang))), (255, 60, 60), 2)
        for ang in r["vein_angles"]:
            cv2.line(m, (R, R), (int(R + 60 * np.cos(ang)), int(R + 60 * np.sin(ang))), (30, 140, 255), 2)

        lvl, _ = classify(r["narrowing"])
        ax[i, 0].imshow(crop)
        ax[i, 0].set_title(
            f"#{i + 1}  Daralma: %{100 * r['narrowing']:.0f} ({lvl})\n"
            f"Salus Sapma Açısı: {r.get('salus_angle_deg', 0.0):.1f}°"
        )
        ax[i, 0].axis("off")

        ax[i, 1].imshow(m)
        ax[i, 1].set_title(
            f"Mavi: Toplardamar | Kırmızı: Atardamar\n"
            f"Uzak: {r['vein_far']:.1f}px → Yakın: {r['vein_near']:.1f}px"
        )
        ax[i, 1].axis("off")

        # 3. Sütun: Kesişimden Uzaklaştıkça Toplardamar Çap Profili W(d)
        d_vals = r.get("profile_d", [])
        w_mean = r.get("profile_w_mean", [])
        if d_vals and w_mean:
            ax[i, 2].plot(d_vals, w_mean, color="#2563eb", linewidth=2.3, label="Toplardamar Çapı W(d)")
            ax[i, 2].axhline(r["vein_far"], color="#16a34a", linestyle="--", linewidth=1.4, label=f"Uzak Çap ({r['vein_far']:.1f}px)")
            ax[i, 2].axhline(r["vein_near"], color="#dc2626", linestyle=":", linewidth=1.4, label=f"Daralmış Çap ({r['vein_near']:.1f}px)")
            ax[i, 2].axvspan(0, 10, color="#fee2e2", alpha=0.5, label="Sıkışma (Nicking) Bölgesi")
            ax[i, 2].set_xlabel("Kesişim Sınırından Mesafe d (px)")
            ax[i, 2].set_ylabel("Toplardamar Çapı (px)")
            ax[i, 2].set_title(f"#{i + 1} Sub-Pixel Toplardamar Çap Profili")
            ax[i, 2].legend(loc="lower right", fontsize=7.5)
            ax[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=105)
    plt.close(fig)
    return True


def main():
    goruntuler = find_fundus_images(DATA_DIR)
    print(f"{len(goruntuler)} fundus görüntüsü analiz edilecek...")
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

    print(f"Tüm sonuçlar kaydedildi: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
