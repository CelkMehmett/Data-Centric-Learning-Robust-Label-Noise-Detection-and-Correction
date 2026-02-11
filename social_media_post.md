# 🚀 Data-Centric AI Projesi: Etiket Gürültüsü Tespiti ve Düzeltilmesi

Merhaba arkadaşlar! 👋

Bugün sizlerle, model mimarisini değiştirmeden sadece veri kalitesini artırarak makine öğrenmesi performansını nasıl iyileştirebileceğimizi gösteren yeni projemi paylaşmak istiyorum: **Data-Centric AI ile Gürültü Tespiti ve Düzeltilmesi**.

🔍 **Projenin Amacı:**
Makine öğrenmesi projelerinde genellikle daha karmaşık modeller kurmaya odaklanılır. Ancak "Çöp giren çöp çıkar" (Garbage In, Garbage Out) prensibi gereği, verimiz kötüyse modelimiz de kötü olacaktır. Bu projede, CIFAR-10 verisetine kontrollü olarak gürültü (yanlış etiketler) ekledim ve ardından **Confident Learning (Güvenli Öğrenme)** tekniklerini kullanarak bu hataları tespit edip düzelttim.

🛠️ **Kullanılan Teknolojiler & Yöntemler:**
*   **PyTorch & ResNet-18**: Temel model eğitimi için.
*   **Cleanlab**: Etiket hatalarını otomatik tespit etmek için.
*   **Aktif Öğrenme (Active Learning)**: Modelin en çok zorlandığı örnekleri seçip düzelterek verimliliği artırmak için.
*   **Gürültü Türleri**: Simetrik (Rastgele) ve Asimetrik (Örn: Kedi -> Köpek karışıklığı) gürültü senaryoları.

📊 **Sonuçlar:**
Proje, modelin sadece veri temizliği yapılarak (mimari değişmeden) %X oranında daha iyi performans gösterebileceğini kanıtlıyor. Ayrıca, hangi veri temizleme stratejisinin (Silme, Yeniden Etiketleme, Ağırlıklandırma) hangi durumda daha etkili olduğunu analiz ettim.

🔗 **Github Deposu:**
Kodları ve detaylı incelemeyi burada bulabilirsiniz: [https://github.com/CelkMehmett/Data-Centric-Learning-Robust-Label-Noise-Detection-and-Correction](https://github.com/CelkMehmett/Data-Centric-Learning-Robust-Label-Noise-Detection-and-Correction)

Bu alanda çalışan veya ilgilenen herkesle fikir alışverişinde bulunmaktan memnuniyet duyarım! Yorumlarınızı ve geri bildirimlerinizi bekliyorum. 👇

#DataScience #MachineLearning #AI #DataCentricAI #Cleanlab #PyTorch #DeepLearning #ArtificialIntelligence
