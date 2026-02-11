import numpy as np
from src.dataset import CIFAR10Noise

def test_noise_injection():
    print("Simetrik Gürültü Enjeksiyonu Test Ediliyor...")
    dataset_sym = CIFAR10Noise(root='./data', train=True, download=True, 
                               noise_type='symmetric', noise_rate=0.2)
    
    clean_targets = dataset_sym.clean_targets
    noisy_targets = dataset_sym.targets
    noise_ratio = np.mean(clean_targets != noisy_targets)
    print(f"İstenen Gürültü Oranı: 0.2")
    print(f"Gerçekleşen Gürültü Oranı: {noise_ratio:.4f}")
    
    if abs(noise_ratio - 0.2) < 0.05:
        print("BAŞARILI: Simetrik gürültü oranı beklenen aralıkta.")
    else:
        print("BAŞARISIZ: Simetrik gürültü oranı önemli ölçüde sapıyor.")

    print("\nAsimetrik Gürültü Enjeksiyonu Test Ediliyor...")
    dataset_asym = CIFAR10Noise(root='./data', train=True, download=True,
                                noise_type='asymmetric', noise_rate=0.4)
    
    clean_targets = dataset_asym.clean_targets
    noisy_targets = dataset_asym.targets
    noise_ratio = np.mean(clean_targets != noisy_targets)
    print(f"İstenen Asimetrik Gürültü Oranı: 0.4 (belirli sınıflarda)")
    print(f"Gerçekleşen Gürültü Oranı (genel): {noise_ratio:.4f}")
    
    # Özel eşlemeleri kontrol et
    # 9 -> 1 (Kamyon -> Otomobil)
    truck_indices = np.where(clean_targets == 9)[0]
    flipped_trucks = np.sum(noisy_targets[truck_indices] == 1)
    total_trucks = len(truck_indices)
    print(f"Otomobile (1) çevrilen Kamyonlar (9): {flipped_trucks}/{total_trucks} ({flipped_trucks/total_trucks:.2f})")
    
    if abs((flipped_trucks/total_trucks) - 0.4) < 0.05:
        print("BAŞARILI: Asimetrik gürültü eşlemesi (Kamyon->Oto) doğru.")
    else:
        print("BAŞARISIZ: Asimetrik gürültü eşlemesi sapıyor.")

if __name__ == "__main__":
    test_noise_injection()
