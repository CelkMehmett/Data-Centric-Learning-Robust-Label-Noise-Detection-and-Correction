from src.train import train_model
import json
import os

def run_all_experiments():
    results = {}
    
    # 1. Temiz Baz Model (Clean Baseline - %0 Gürültü)
    print("\n--- Temiz Baz Model (Clean Baseline) Çalıştırılıyor ---")
    acc, f1, cm = train_model(noise_type=None, noise_rate=0.0, num_epochs=15)
    results['clean'] = {'acc': acc, 'f1': f1, 'cm': cm.tolist()}
    
    # 2. Simetrik Gürültü (%20) - Yaygın bir kıyaslama
    print("\n--- Simetrik Gürültü (%20) Çalıştırılıyor ---")
    acc, f1, cm = train_model(noise_type='symmetric', noise_rate=0.2, num_epochs=15)
    results['symmetric_0.2'] = {'acc': acc, 'f1': f1, 'cm': cm.tolist()}
    
    # 3. Asimetrik Gürültü (%40) - Daha zor
    print("\n--- Asimetrik Gürültü (%40) Çalıştırılıyor ---")
    acc, f1, cm = train_model(noise_type='asymmetric', noise_rate=0.4, num_epochs=15)
    results['asymmetric_0.4'] = {'acc': acc, 'f1': f1, 'cm': cm.tolist()}
    
    # Sonuçları kaydet
    with open('baseline_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nTüm deneyler tamamlandı. Sonuçlar baseline_results.json dosyasına kaydedildi.")

if __name__ == "__main__":
    run_all_experiments()
