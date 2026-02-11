# 🚀 Veri Odaklı Yapay Zeka (Data-Centric AI): %81.5 Başarı ile Gürültü Tespiti!

**Model mimarisini değiştirmeden model performansını artırmak mümkün mü? Evet, veriyi temizleyerek!**

Merhaba, bugün sizlerle yeni projem **"Data-Centric Learning: Robust Label Noise Detection"**ı paylaşmaktan heyecan duyuyorum. Makine öğrenmesinde genellikle daha karmaşık modeller kurmaya odaklanılır, ancak "Çöp giren çöp çıkar" (Garbage In, Garbage Out) prensibi gereği, verimiz kötüyse en iyi model bile başarısız olacaktır.

Bu projede, bu sorunu ele alarak CIFAR-10 verisetini kontrollü olarak bozdum (%20 Simetrik Gürültü), analiz ettim ve temizledim.

🔍 **Neler Yaptım ve Başardım?**
1.  **Gürültü Enjeksiyonu**: Veriye %20 oranında hatalı etiketler ekledim.
2.  **Yüksek Tespit Başarısı**: `cleanlab` ve Güvenli Öğrenme (Confident Learning) kullanarak, sadece 2 epoch'luk hızlı eğitimde bile **eklenen gürültünün %81.5'ini başarıyla tespit ettim (Recall)!**
3.  **İleri Görselleştirme (t-SNE)**: Hatalı etiketlerin özellik uzayında (feature space) nasıl davrandığını ve temiz kümelerin içine nasıl sızdığını 2 Boyutlu haritalarla görselleştirdim.
4.  **Temizleme Stratejileri**: Hatalı verileri sadece silmekle (Drop) kalmadım; onları model uzlaşısı ile **Yeniden Etiketleme (Relabel)** yöntemiyle kurtardım.

🛠️ **Kullanılan Teknolojiler:**
*   **PyTorch & ResNet-18**
*   **Cleanlab** (Gürültü Tespiti)
*   **t-SNE & Matplotlib** (Veri Görselleştirme)

📊 **Neden Önemli?**
Genellikle veriyi temizlemek, model hiperparametrelerini ayarlamaktan çok daha büyük getiri sağlar. Bu proje, kirli verilerle çalışırken bile basit ama etkili stratejilerle modelin nasıl sağlam (robust) hale getirilebileceğini gösteriyor.

🔗 **Github Deposu ve Detaylı Rapor:**
Kodlara, analiz grafiklerine ve sonuç raporuna buradan ulaşabilirsiniz:
👉 [https://github.com/CelkMehmett/Data-Centric-Learning-Robust-Label-Noise-Detection-and-Correction](https://github.com/CelkMehmett/Data-Centric-Learning-Robust-Label-Noise-Detection-and-Correction)

Veri kalitesi ve Data-Centric AI alanında çalışan herkesin yorumlarını ve katkılarını bekliyorum! 👇

#DataScience #MachineLearning #AI #DataCentricAI #DeepLearning #PyTorch #Cleanlab #DataQuality #BigData #ArtificialIntelligence
