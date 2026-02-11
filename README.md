# Data-Centric AI: CIFAR-10 Gürültü Tespiti ve Kurtarma

Bu proje, model mimarisini değiştirmek yerine etiket hatalarını tespit edip düzelterek model performansını artırmaya yönelik Data-Centric AI (Veri Odaklı YZ) yaklaşımını gösterir.

## 🚀 Özellikler
- **Gürültü Ekleme**: Simetrik (rastgele) ve Asimetrik (sınıf-bağımlı) etiket gürültüsünün kontrollü enjeksiyonu.
- **Baz Modelleme**: Gürültülü veriler üzerinde ResNet-18 eğitimi.
- **Gürültü Tespiti**: Etiket hatalarını belirlemek için `cleanlab` (Güvenli Öğrenme) kullanır.
- **Veri Temizleme**: Silme (Drop), Yeniden Etiketleme (Relabel) ve Yeniden Ağırlıklandırma (Reweight) stratejilerini uygular.
- **Aktif Öğrenme**: Belirsizlik Örneklemesi (Uncertainty Sampling) kullanarak etiketleri yinelemeli olarak düzelten döngü.

## 📂 Yapı
- `src/`: Veriseti, model, eğitim ve temizleme mantığı için kaynak kodları.
- `notebooks/`: Demo scriptleri ve görselleştirme.
- `data/`: Veriseti depolama alanı.

## 🛠 Kullanım

### 1. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 2. Tam İşlem Hattını Çalıştır (Pipeline)
Tüm iş akışını çalıştırmak için (Baz Model -> Tespit -> Temizleme):
```bash
# Hızlı çalıştırma (hata ayıklama için)
python3 run_pipeline.py --noise_type symmetric --noise_rate 0.2 --quick

# Tam deney (Simetrik %20)
python3 run_pipeline.py --noise_type symmetric --noise_rate 0.2

# Tam deney (Asimetrik %40)
python3 run_pipeline.py --noise_type asymmetric --noise_rate 0.4
```

### 3. Bireysel Bileşenler
**Gürültü Enjeksiyonunu Doğrula:**
```bash
python3 verify_noise.py
```

**Baz Model Deneylerini Çalıştır:**
```bash
python3 run_experiments.py
```

**Gürültü Tespitini Çalıştır:**
```bash
python3 run_detection.py
```

**Aktif Öğrenme Simülasyonu:**
```bash
python3 src/active_learning.py
```

## 📊 Sonuçlar
Sonuçlar, aşağıdakileri içeren JSON dosyaları (örn. `pipeline_results_symmetric_0.2.json`) olarak kaydedilir:
- Baz Model Doğruluk & F1
- Tespit Kesinliği & Duyarlılığı (Precision & Recall)
