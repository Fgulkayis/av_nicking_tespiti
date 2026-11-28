import os
import json
import tempfile
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.av_nicking import run_pipeline, text_report, classify, PROJE_KOKU, MODELS_DIR
from main import find_fundus_images, DATA_DIR, RESULTS_DIR

st.set_page_config(
    page_title="Retinal AV Nicking ve Hipertansif Retinopati Erken Teşhis Sistemi",
    page_icon="👁️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_eval_metrics():
    json_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def cached_run_pipeline(img_path, mtime):
    return run_pipeline(img_path)


def render_crossing_row(idx, r, res, R=70):
    cy, cx = r["center"]
    pad = lambda a: np.pad(a, ((R, R), (R, R)) + ((0, 0),) * (a.ndim - 2))
    crop = pad(res["rgb"])[cy:cy + 2 * R, cx:cx + 2 * R]
    m = (pad(res["mask"])[cy:cy + 2 * R, cx:cx + 2 * R] * 255).astype(np.uint8)
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
    m[pad(res["skel"])[cy:cy + 2 * R, cx:cx + 2 * R]] = [255, 220, 0]

    for ang in r.get("art_angles", ()):
        cv2.line(m, (R, R), (int(R + 58 * np.cos(ang)), int(R + 58 * np.sin(ang))), (255, 65, 65), 2)
    for ang in r["vein_angles"]:
        cv2.line(m, (R, R), (int(R + 60 * np.cos(ang)), int(R + 60 * np.sin(ang))), (40, 140, 255), 2)

    lvl, _ = classify(r["narrowing"])
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    ax[0].imshow(crop)
    ax[0].set_title(f"#{idx} Yakın Görünüm | Daralma: %{100*r['narrowing']:.0f} ({lvl})", fontsize=10.5, fontweight="bold")
    ax[0].axis("off")

    ax[1].imshow(m)
    ax[1].set_title(
        f"Mavi: Toplardamar | Kırmızı: Atardamar\n"
        f"Uzak: {r['vein_far']:.1f}px → Yakın: {r['vein_near']:.1f}px | Salus: {r.get('salus_angle_deg', 0):.1f}°",
        fontsize=10,
    )
    ax[1].axis("off")

    d_vals = r.get("profile_d", [])
    w_mean = r.get("profile_w_mean", [])
    if d_vals and w_mean:
        ax[2].plot(d_vals, w_mean, color="#2563eb", linewidth=2.4, label="Toplardamar Çapı W(d)")
        ax[2].axhline(r["vein_far"], color="#16a34a", linestyle="--", linewidth=1.4, label=f"Normal Uzak Çap ({r['vein_far']:.1f}px)")
        ax[2].axhline(r["vein_near"], color="#dc2626", linestyle=":", linewidth=1.4, label=f"Sıkışmış Çap ({r['vein_near']:.1f}px)")
        ax[2].axvspan(0, 10, color="#fee2e2", alpha=0.55, label="AV Nicking (Ezilme) Bölgesi")
        ax[2].set_xlabel("Kesişim Sınırından Mesafe d (piksel)")
        ax[2].set_ylabel("Toplardamar Çapı (px)")
        ax[2].set_title(f"#{idx} Piksel-Altı (Sub-Pixel) Çap Değişim Grafiği", fontsize=10.5, fontweight="bold")
        ax[2].legend(loc="lower right", fontsize=8)
        ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    st.title("👁️ Retinal AV Nicking ve Hipertansif Retinopati Erken Teşhis Sistemi")
    st.markdown(
        "**Hibrit Mikrovasküler Analiz Motoru:** Yalnızca ham göz dibi (fundus) fotoğrafından "
        "Çok Ölçekli Frangi + Top-Hat Segmentasyonu, Makine Öğrenmesi Tabanlı Atardamar/Toplardamar (A/V) Ayrımı, "
        "Piksel-Altı (Sub-Pixel) Dikey Kesit Çap Ölçümü ve 4 Klinik Biyobelirteç "
        "(**Gunn Daralması, Salus Sapma Açısı, AVR Oranı, Damar Kıvrımlılığı**) ile dinamik erken teşhis yapar."
    )

    eval_data = load_eval_metrics()

    # Sol Menü: Görüntü Seçimi veya Yükleme (Statik hiçbir veri kullanılmaz)
    st.sidebar.header("📂 1. Göz Fotoğrafı Girişi")
    kaynak = st.sidebar.radio(
        "Görüntü Kaynağı:",
        ["Bilgisayardan Yeni Fotoğraf Yükle (Canlı Muayene)", "Hazır Test Görüntülerinden Seç"],
        index=0,
    )

    secilen_yol = None
    secilen_ad = None

    if kaynak == "Bilgisayardan Yeni Fotoğraf Yükle (Canlı Muayene)":
        yuklenen = st.sidebar.file_uploader(
            "Retina Fundus Fotoğrafı Yükleyin (.png, .jpg, .tif)",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        )
        if yuklenen is not None:
            suffix = os.path.splitext(yuklenen.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(yuklenen.getbuffer())
                secilen_yol = tmp.name
                secilen_ad = yuklenen.name
    else:
        mevcut_goruntuler = find_fundus_images(DATA_DIR)
        if not mevcut_goruntuler:
            st.sidebar.warning("`data/` klasöründe görüntü bulunamadı.")
        else:
            etiketler = []
            yol_map = {}
            for p in mevcut_goruntuler:
                fname = os.path.basename(p)
                ident = os.path.splitext(fname)[0]
                lbl = f"Fundus Görüntüsü #{ident} ({fname})"
                etiketler.append(lbl)
                yol_map[lbl] = (p, fname)

            secim = st.sidebar.selectbox("Test Görüntüsü Seçin:", etiketler)
            secilen_yol, secilen_ad = yol_map[secim]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Ölçülen 4 Biyobelirteç")
    st.sidebar.markdown(
        "1. **Gunn Belirtisi (AV Nicking):** Sertleşen atardamarın toplardamarı kesişimde ezerek daraltması.\n"
        "2. **Salus Belirtisi (Sapma Açısı):** Ezilen toplardamarın kesişimde **S** harfi çizerek açısal sapması.\n"
        "3. **AVR (Çap Oranı):** Atardamarların genel incelmesi (`Normal >= 0.67`).\n"
        "4. **Tortuosity (Kıvrımlılık):** Yüksek basınç nedeniyle damarların yılan gibi kıvrımlılaşması."
    )

    tab1, tab2, tab3 = st.tabs([
        "🔬 1. Canlı Retina Analizi ve Erken Teşhis",
        "📊 2. Eğitilen Yapay Zeka Modelleri ve Başarı Metrikleri",
        "📋 3. Veri Seti Doğrulama Tablosu",
    ])

    # =========================================================================
    # SEKME 1: CANLI RETİNA ANALİZİ VE ERKEN TEŞHİS
    # =========================================================================
    with tab1:
        if secilen_yol is None or not os.path.exists(secilen_yol):
            st.info(
                "👈 Analize başlamak için sol menüden **'Browse files'** butonuna tıklayarak bir göz fotoğrafı yükleyin "
                "(Örn: `sunum_test_goruntuleri/1_modele_verilecek_ham_goruntu.png`) veya hazır test görüntülerinden birini seçin."
            )
        else:
            with st.spinner(f"`{secilen_ad}` yüksek çözünürlükte analiz ediliyor..."):
                mtime = os.path.getmtime(secilen_yol)
                res = cached_run_pipeline(secilen_yol, mtime)

            diag = res["diagnosis"]
            bm = diag["biomarkers"]
            rs = res["results"]

            risk_pct = diag["risk_Percentage"]

            # Üst Karar Banner'ı
            if diag["badge_color"] == "red":
                st.error(f"### 🚨 Erken Teşhis Sonucu: **{diag['risk_level']}**  —  Risk Skoru: **%{risk_pct:.1f}**")
            elif diag["badge_color"] == "orange":
                st.warning(f"### ⚠️ Erken Teşhis Sonucu: **{diag['risk_level']}**  —  Risk Skoru: **%{risk_pct:.1f}**")
            else:
                st.success(f"### ✅ Erken Teşhis Sonucu: **{diag['risk_level']}**  —  Risk Skoru: **%{risk_pct:.1f}**")

            # KPI Kartları (%100 görüntüden dinamik hesaplanan değerler)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Erken Teşhis Risk Skoru", f"%{risk_pct:.1f}", f"{bm['high_nicking_count']} Yüksek Riskli Kesişim")
            c2.metric("Maks. Gunn Daralması", f"%{bm['max_narrowing']*100:.0f}", f"{len(rs)} A/V Çaprazlaşma")
            c3.metric("Maks. Salus Sapma Açısı", f"{bm['max_salus_deg']:.1f}°", f"Ort: {bm['mean_salus_deg']:.1f}°")
            c4.metric("AVR (Atar/Toplar Çapı)", f"{bm['avr']:.2f}", "Normal >= 0.67")
            c5.metric("Damar Kıvrımlılığı (Tortuosity)", f"{bm['tortuosity_mean']:.3f}", f"80. Persentil: {bm['tortuosity_p80']:.3f}")

            st.markdown("---")

            # 4'lü Görsel Harita Paneli
            st.subheader("🖼️ Adım Adım Görüntü İşleme ve Yapay Zeka Haritaları")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.image(res["rgb"], caption="1) Orijinal Fundus Görüntüsü", use_container_width=True)
            with g2:
                v_map = np.dstack([res["mask"] * 95] * 3).astype(np.uint8)
                v_map[res["skel"]] = [255, 255, 0]
                st.image(v_map, caption=f"2) Hibrit Damar Maskesi ve İskelet (Kıvrımlılık: {bm['tortuosity_mean']:.3f})", use_container_width=True)
            with g3:
                st.image(res["av_info"]["av_vis"], caption=f"3) Model 1 A/V Haritası (Kırmızı: Atar, Mavi: Toplar | AVR: {bm['avr']:.2f})", use_container_width=True)
            with g4:
                st.image(res["annotated"], caption=f"4) Tespit Edilen Çaprazlaşmalar ({len(rs)} Aday | Mavi: Optik Disk)", use_container_width=True)

            # Klinik Bulgular ve 4 Biyobelirteç Skor Dağılımı
            col_left, col_right = st.columns([1.25, 1.0])
            with col_left:
                st.subheader("🩺 Otomatik Klinik Bulgular ve Açıklanabilir Karar Özeti")
                for finding in diag["clinical_findings"]:
                    st.markdown(f"- {finding}")

                st.markdown("#### 📌 Tespit Edilen A/V Çaprazlaşma Noktaları Özeti")
                if rs:
                    tablo_verisi = []
                    for i, r in enumerate(rs, 1):
                        lvl, _ = classify(r["narrowing"])
                        av_conf = 0.5 * (r.get("artery_prob", 0.5) + r.get("vein_prob", 0.5))
                        tablo_verisi.append({
                            "Kesişim No": f"#{i}",
                            "Konum (x, y)": f"({r['center'][1]}, {r['center'][0]})",
                            "Toplardamar Uzak Çap": f"{r['vein_far']:.1f} px",
                            "Toplardamar Yakın Çap": f"{r['vein_near']:.1f} px",
                            "Gunn Daralma (%)": f"%{100*r['narrowing']:.0f}",
                            "Salus Sapma Açısı": f"{r.get('salus_angle_deg', 0):.1f}°",
                            "A/V Model Güveni": f"%{100*av_conf:.0f}",
                            "Risk Seviyesi": lvl,
                        })
                    st.dataframe(tablo_verisi, use_container_width=True, hide_index=True)
                else:
                    st.info("Optik disk dışında belirgin bir ana damar çaprazlaşması bulunamadı.")

            with col_right:
                st.subheader("📈 4 Biyobelirteç Risk Katkı Analizi")
                comp = diag["component_scores"]
                fig_bar, ax_bar = plt.subplots(figsize=(6.2, 3.8))
                labels_b = list(comp.keys())
                vals_b = list(comp.values())
                colors_b = ["#ef4444" if v >= 60 else ("#f59e0b" if v >= 35 else "#22c55e") for v in vals_b]
                bars = ax_bar.barh(labels_b[::-1], vals_b[::-1], color=colors_b[::-1], height=0.55)
                ax_bar.set_xlim(0, 108)
                ax_bar.set_xlabel("Biyobelirteç Anomali Skoru (0 - 100)")
                ax_bar.grid(True, axis="x", alpha=0.3)
                for bar in bars:
                    w_val = bar.get_width()
                    ax_bar.annotate(
                        f"%{w_val:.1f}",
                        xy=(w_val, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontweight="bold",
                    )
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            # Çaprazlaşmaların Yakın Görünümü ve Sub-Pixel Çap Değişim Grafikleri
            if rs:
                st.markdown("---")
                st.subheader("🔍 Çaprazlaşma Noktalarının Yakın İncelemesi ve Piksel-Altı (Sub-Pixel) Çap Profilleri")
                st.caption(
                    "Her kesişim için sol panelde orijinal yakın görünüm, orta panelde Atardamar (Kırmızı) ve Toplardamar (Mavi) doğrultuları, "
                    "sağ panelde ise toplardamarın kesişim noktasına yaklaştıkça nasıl ezilip inceldiğini gösteren çap eğrisi W(d) yer alır."
                )
                gosterilecek_sayi = st.slider(
                    "Gösterilecek Çaprazlaşma Sayısı (En yüksek daralmadan başlayarak):",
                    min_value=1,
                    max_value=len(rs),
                    value=min(5, len(rs)),
                )
                for idx in range(gosterilecek_sayi):
                    fig_c = render_crossing_row(idx + 1, rs[idx], res)
                    st.pyplot(fig_c)
                    plt.close(fig_c)

    # =========================================================================
    # SEKME 2: EĞİTİLEN YAPAY ZEKA MODELLERİ VE BAŞARI METRİKLERİ
    # =========================================================================
    with tab2:
        st.subheader("🤖 Eğitilen 3 Makine Öğrenmesi Modeli ve 5-Katlı Çapraz Doğrulama (5-Fold CV) Sonuçları")
        if eval_data is None:
            st.warning("Model değerlendirme dosyası (`models/evaluation_metrics.json`) henüz oluşturulmamış. `py src/train_eval.py` komutunu çalıştırın.")
        else:
            m1 = eval_data["model1_av_classifier"]
            m2 = eval_data["model2_crossing_validator"]
            m3 = eval_data["model3_early_diagnosis"]

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown("#### 1️⃣ Model 1: Atardamar / Toplardamar Ayrımı")
                st.caption(f"Algoritma: ExtraTreesClassifier | Örneklem: {m1['sample_count']} damar segmenti")
                st.metric("Doğruluk (Accuracy)", f"%{m1['accuracy']}")
                st.metric("F1-Skoru", f"%{m1['f1_score']}")
                st.metric("ROC-AUC Skoru", f"%{m1['roc_auc']}")

            with col_m2:
                st.markdown("#### 2️⃣ Model 2: A/V Kesişim Doğrulayıcı")
                st.caption(f"Algoritma: RandomForestClassifier | Örneklem: {m2['sample_count']} düğüm noktası")
                st.metric("Doğruluk (Accuracy)", f"%{m2['accuracy']}")
                st.metric("F1-Skoru", f"%{m2['f1_score']}")
                st.metric("ROC-AUC Skoru", f"%{m2['roc_auc']}")

            with col_m3:
                st.markdown("#### 3️⃣ Model 3: Erken Teşhis Risk Modeli")
                st.caption(f"Algoritma: GradientBoostingClassifier | Örneklem: {m3['sample_count']} klinik/bölgesel örnek")
                st.metric("Doğruluk (Accuracy)", f"%{m3['accuracy']}")
                st.metric("F1-Skoru", f"%{m3['f1_score']}")
                st.metric("ROC-AUC Skoru", f"%{m3['roc_auc']}")

            with st.expander("🏗️ Hibrit Sistem Mimarisi ve Modüllerin Görev Tablosu (Açmak / Kapatmak İçin Tıklayın)", expanded=False):
                mimari_tablosu = [
                    {
                        "Modül / Aşama": "1. Ön İşleme ve Hibrit Damar Segmentasyonu",
                        "Kullanılan Yöntem / Algoritma": "Yeşil Kanal CLAHE + 5 Ölçekli Black Top-Hat + Frangi Tübülerlik + Çift Eşikli Histerezis",
                        "Klinik Amacı": "Sıkışmış ve kontrastı düşmüş ince damarları kesişim noktasında koparmadan bütüncül çıkarmak",
                    },
                    {
                        "Modül / Aşama": "2. Model 1: Atardamar / Toplardamar (A/V) Ayrımı",
                        "Kullanılan Yöntem / Algoritma": f"14 Renk (RGB, HSV, CIE LAB) ve Merkez Yansıma (CLR) Özniteliği + ExtraTrees (%{m1['accuracy']} Doğruluk, %{m1['roc_auc']} AUC)",
                        "Klinik Amacı": "Üstten geçen sertleşmiş atardamar ile altta ezilen toplardamarı otomatik ayırt etmek",
                    },
                    {
                        "Modül / Aşama": "3. Model 2: A/V Çaprazlaşma Doğrulayıcı",
                        "Kullanılan Yöntem / Algoritma": f"9 Geometrik ve Optik Fark Özniteliği + RandomForest (%{m2['accuracy']} Doğruluk, %{m2['roc_auc']} AUC)",
                        "Klinik Amacı": "Gerçek atardamar-toplardamar kesişimlerini tek damar çatallanmalarından (dallanmalardan) ayırmak",
                    },
                    {
                        "Modül / Aşama": "4. Piksel-Altı (Sub-Pixel) Çap ve Açı Ölçümü",
                        "Kullanılan Yöntem / Algoritma": "Atardamar Koridoru Maskeli Ortogonal (Dikey) Kesit FWHM Çap Profili W(d) ve Yörünge Takibi",
                        "Klinik Amacı": "Gunn daralma yüzdesini, venöz birikimi (banking) ve Salus S-sapma açısını milimetrik ölçmek",
                    },
                    {
                        "Modül / Aşama": "5. Model 3: Erken Teşhis Risk Sınıflandırıcı",
                        "Kullanılan Yöntem / Algoritma": f"4 Klinik Biyobelirteç Ailesi (Gunn + Salus + AVR + Kıvrımlılık) + GradientBoosting (%{m3['accuracy']} Doğruluk, %{m3['roc_auc']} AUC)",
                        "Klinik Amacı": "Hastanın erken evre hipertansif retinopati riskini açıklanabilir yapay zeka (XAI) ile skorlamak",
                    },
                ]
                st.table(mimari_tablosu)

            plot_path = os.path.join(RESULTS_DIR, "model_degerlendirme_grafikleri.png")
            if os.path.exists(plot_path):
                st.subheader("📈 ROC Eğrileri, Performans Metrikleri ve Öznitelik Önem Düzeyleri (XAI)")
                st.image(plot_path, use_container_width=True)

    # =========================================================================
    # SEKME 3: KLİNİK VERİ SETİ (LES-AV) DOĞRULAMA TABLOSU
    # =========================================================================
    with tab3:
        st.subheader("📋 Model Eğitiminde ve Doğrulamasında Kullanılan LES-AV Veri Seti Özeti")
        st.caption("Not: Bu tablo yalnızca modelin eğitildiği veri setinin klinik referans doğrulaması içindir; canlı analiz sekmesinde bu tablodaki hiçbir statik veri kullanılmaz.")
        if eval_data and "patients" in eval_data:
            st.dataframe(eval_data["patients"], use_container_width=True, hide_index=True)
        else:
            st.info("Hasta tablosunu görmek için model eğitim betiğini çalıştırın.")


if __name__ == "__main__":
    main()
