from src.clean import get_out_of_sample_probs
from src.dataset import CIFAR10Noise, get_transforms
from cleanlab.filter import find_label_issues
import numpy as np
import json

def evaluate_detection(noise_type, noise_rate):
    print(f"\n--- {noise_type} ({noise_rate}) için Gürültü Tespiti Çalıştırılıyor ---")
    
    # Verisetini yükle
    dataset = CIFAR10Noise(root='./data', train=True, download=True,
                           transform=get_transforms(train=True),
                           noise_type=noise_type, noise_rate=noise_rate)
    
    # Gerçek Doğruluk (Ground Truth): Hangi indeksler gerçekten çevrildi?
    # Bunu clean_targets ve targets karşılaştırarak anlayabiliriz
    actual_noise_indices = np.where(dataset.targets != dataset.clean_targets)[0]
    print(f"Gerçek Gürültülü Örnekler: {len(actual_noise_indices)} / {len(dataset)}")
    
    # CV ile Örnek Dışı Olasılıkları (Out-of-Sample Probabilities) Al
    # Demo için epoch sayısını düşük tutuyoruz, ideal olarak 10+
    psx = get_out_of_sample_probs(dataset, k=3, epochs=3) 
    
    # Sorunları bul
    # cleanlab bir boolean maskesi döndürür (True = etiket sorunu)
    issues_mask = find_label_issues(
        labels=dataset.targets,
        pred_probs=psx,
        return_indices_ranked_by=None # Boolean maskesi döndür
    )
    
    detected_indices = np.where(issues_mask)[0]
    print(f"Tespit Edilen Gürültülü Örnekler: {len(detected_indices)}")
    
    # Tespit için Kesinlik/Duyarlılık/F1 Hesapla (Precision/Recall/F1)
    # Doğru Pozitifler (TP): Hem gerçek gürültüde hem de tespit edilenlerde olan indeksler
    tp = len(np.intersect1d(actual_noise_indices, detected_indices))
    fp = len(np.setdiff1d(detected_indices, actual_noise_indices))
    fn = len(np.setdiff1d(actual_noise_indices, detected_indices))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Tespit Performansı:")
    print(f"  Kesinlik (Precision): {precision:.4f}")
    print(f"  Duyarlılık (Recall):  {recall:.4f}")
    print(f"  F1 Skoru:             {f1:.4f}")
    
    # Sorunları daha sonra kullanmak üzere kaydet (Veri Temizleme adımı)
    results = {
        'noise_type': noise_type,
        'noise_rate': noise_rate,
        'detected_indices': detected_indices.tolist(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    with open(f'detection_results_{noise_type}_{noise_rate}.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    # Simetrik %20 için çalıştır
    evaluate_detection('symmetric', 0.2)
