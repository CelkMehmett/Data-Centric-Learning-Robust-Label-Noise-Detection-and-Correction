# Data-Centric AI: CIFAR-10 Gürültü Tespiti ve Kurtarma

Bu proje, model mimarisini değiştirmek yerine etiket hatalarını tespit edip düzelterek model performansını artırmaya yönelik Data-Centric AI (Veri Odaklı YZ) yaklaşımını gösterir.

## 🚀 Özellikler (Data-Centric Yaklaşımı)
- **Gürültü Ekleme**: Simetrik (rastgele) ve Asimetrik (sınıf-bağımlı) etiket gürültüsünün kontrollü enjeksiyonu.
- **İleri Veri Analizi**:
    - **t-SNE**: Gürültülü ve temiz örneklerin özellik uzayındaki görsel dağılımı.
    - **Kayıp Analizi (Loss Analysis)**: Modelin hatalı etiketlere verdiği tepkinin histogram analizi.
- **Gürültü Tespiti**: Etiket hatalarını belirlemek için `cleanlab` (Güvenli Öğrenme) kullanır.
- **Veri Temizleme**: Silme (Drop), Yeniden Etiketleme (Relabel) ve Yeniden Ağırlıklandırma (Reweight) stratejilerini uygular.

## 📂 Yapı
- `src/`: Veriseti, model, eğitim, temizleme ve **analiz** kodları.
- `notebooks/`: Demo scriptleri.
- `tests/`: Doğrulama testleri.
- `report_images/`: Oluşturulan analiz grafikleri (t-SNE, Loss vb.).

## 🛠 Kullanım

### 1. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 2. Tam İşlem Hattını Çalıştır (Pipeline)
Bu komut sırasıyla:
1. Baz modeli eğitir.
2. t-SNE ve Kayıp grafiklerini oluşturur.
3. Etiket hatalarını tespit eder.
4. Temizleme stratejilerini uygular.

```bash
# Hızlı test (Demo modu)
python3 run_pipeline.py --noise_type symmetric --noise_rate 0.2 --quick

# Tam deney (Simetrik %20)
python3 run_pipeline.py --noise_type symmetric --noise_rate 0.2
```

### 3. Raporlama
Sonuçları ve grafikleri içeren HTML raporunu oluşturun:
```bash
python3 generate_report.py
```
Bu işlem `report.html` dosyasını oluşturur. Tarayıcınızda açarak interaktif sonuçları inceleyebilirsiniz.

### 4. Testler
```bash
python3 tests/verify_noise.py
```

## 📊 Çıktılar
- `pipeline_results_*.json`: Sayısal metrikler.
- `report_images/tsne.png`: t-SNE görselleştirmesi.
- `report_images/loss_dist.png`: Kayıp dağılımı histogramı.
- `report.html`: Tüm sonuçların özetlendiği görsel rapor.
