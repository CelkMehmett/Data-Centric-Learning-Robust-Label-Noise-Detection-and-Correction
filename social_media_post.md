# 🚀 Veri Odaklı Yapay Zeka (Data-Centric AI): Gürültü Tespiti ve Kurtarma

**Model mimarisini değiştirmeden model performansını artırmak mümkün mü? Evet, veriyi temizleyerek!**

Merhaba, bugün sizlerle yeni projem **"Data-Centric Learning: Robust Label Noise Detection"**ı paylaşmaktan heyecan duyuyorum. Makine öğrenmesinde genellikle daha karmaşık modeller kurmaya odaklanılır, ancak "Çöp giren çöp çıkar" (Garbage In, Garbage Out) prensibi gereği, verimiz kötüyse en iyi model bile başarısız olacaktır.

Bu projede, bu sorunu ele alarak CIFAR-10 verisetini kontrollü olarak bozdum, analiz ettim ve temizledim. İşte detaylar:

🔍 **Neler Yaptım?**
1.  **Gürültü Enjeksiyonu**: Veriye %20-%40 oranında hatalı etiketler ekledim (Simetrik ve Sınıf-Bağımlı gürültü).
2.  **İleri Görselleştirme (t-SNE)**: Hatalı etiketlerin özellik uzayında (feature space) nasıl davrandığını ve temiz kümelerin içine nasıl sızdığını 2 Boyutlu haritalarla görselleştirdim.
3.  **Kayıp Analizi (Loss Analysis)**: Modelin hatalı örneklere verdiği tepkiyi (daha yüksek eğitim kaybı / forgetting events) histogramlarla kanıtladım.
4.  **Güvenli Öğrenme (Confident Learning)**: `cleanlab` kütüphanesini kullanarak etiket hatalarını otomatik tespit ettim.
5.  **Temizleme Stratejileri**: Hatalı verileri sadece silmekle (Drop) kalmadım; onları model uzlaşısı ile **Yeniden Etiketleme (Relabel)** ve **Yeniden Ağırlıklandırma (Reweight)** yöntemleriyle kurtarmayı denedim.

🛠️ **Kullanılan Teknolojiler:**
*   **PyTorch & ResNet-18**
*   **Cleanlab** (Gürültü Tespiti)
*   **t-SNE & Matplotlib** (Veri Görselleştirme)
*   **Data-Centric AI Prensipleri**

📊 **Sonuçlar:**
Sadece veriyi temizleyerek (model mimarisine dokunmadan) belirgin bir performans artışı sağladık. Özellikle **Yeniden Etiketleme (Relabel)** stratejisinin, veriyi silmeye kıyasla daha fazla bilgi koruduğunu gözlemledim.

🔗 **Github Deposu ve Detaylı Rapor:**
Kodlara, analiz grafiklerine ve sonuç raporuna buradan ulaşabilirsiniz:
👉 [https://github.com/CelkMehmett/Data-Centric-Learning-Robust-Label-Noise-Detection-and-Correction](https://github.com/CelkMehmett/Data-Centric-Learning-Robust-Label-Noise-Detection-and-Correction)

Veri kalitesi ve Data-Centric AI alanında çalışan herkesin yorumlarını ve katkılarını bekliyorum! 👇

#DataScience #MachineLearning #AI #DataCentricAI #DeepLearning #PyTorch #Cleanlab #DataQuality #BigData #ArtificialIntelligence
