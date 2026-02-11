import argparse
import json
import os
import torch
from src.train import train_model
from src.clean import get_out_of_sample_probs
from src.dataset import CIFAR10Noise, get_transforms
from src.cleaning_strategies import apply_drop_strategy, apply_relabel_strategy, apply_reweight_strategy
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

from src.analysis import extract_features_and_loss, plot_tsne, plot_loss_distribution
import os

def run_pipeline(noise_type='symmetric', noise_rate=0.2, quick_mode=False):
    print(f"============================================================")
    print(f"DATA-CENTRIC AI PIPELINE ÇALIŞTIRILIYOR: {noise_type} ({noise_rate})")
    print(f"============================================================")
    
    epochs = 2 if quick_mode else 15
    cv_k = 2 if quick_mode else 4
    
    # Rapor görselleri için klasör
    os.makedirs('report_images', exist_ok=True)
    
    results = {}

    # 1. BAZ MODEL (BASELINE)
    print("\n[ADIM 1] Gürültülü Veri Üzerinde Baz Model Eğitiliyor...")
    model, acc_base, f1_base, cm_base = train_model(noise_type=noise_type, noise_rate=noise_rate, num_epochs=epochs)
    results['baseline'] = {'acc': acc_base, 'f1': f1_base}
    print(f"Baz Model Doğruluğu: {acc_base:.4f}")
    
    # 1.1 İLERİ ANALİZ (GÖRSELLEŞTİRME)
    print("\n[ADIM 1.1] İleri Veri Analizi (t-SNE & Kayıp Dağılımı)...")
    analysis_dataset = CIFAR10Noise(root='./data', train=True, download=True,
                                    noise_type=noise_type, noise_rate=noise_rate,
                                    transform=get_transforms(train=True)) # Eğitim seti üzerinde analiz
    
    features, losses, targets, clean_targets = extract_features_and_loss(model, analysis_dataset)
    
    # t-SNE Çizimi
    plot_tsne(features, targets, clean_targets, 
              title=f"t-SNE: {noise_type} ({noise_rate}) Gürültü", 
              save_path="report_images/tsne.png")
              
    # Kayıp Dağılımı
    plot_loss_distribution(losses, targets, clean_targets, 
                           save_path="report_images/loss_dist.png")

    # 2. GÜRÜLTÜ TESPİTİ
    print("\n[ADIM 2] Cleanlab ile Etiket Hataları Tespit Ediliyor...")
    dataset = CIFAR10Noise(root='./data', train=True, download=True,
                           noise_type=noise_type, noise_rate=noise_rate,
                           transform=get_transforms(train=True))
    
    # Örnek Dışı Olasılıkları Al
    psx = get_out_of_sample_probs(dataset, k=cv_k, epochs=epochs)
    
    # Sorunları Bul
    issues_mask = find_label_issues(labels=dataset.targets, pred_probs=psx)
    num_issues = np.sum(issues_mask)
    print(f"{num_issues} etiket hatası tespit edildi.")
    
    # Tespitin Değerlendirilmesi
    actual_noise_indices = np.where(dataset.targets != dataset.clean_targets)[0]
    detected_indices = np.where(issues_mask)[0]
    tp = len(np.intersect1d(actual_noise_indices, detected_indices))
    precision = tp / len(detected_indices) if len(detected_indices) > 0 else 0
    recall = tp / len(actual_noise_indices) if len(actual_noise_indices) > 0 else 0
    print(f"Tespit Kesinliği (Precision): {precision:.4f}, Duyarlılık (Recall): {recall:.4f}")
    results['detection'] = {'num_issues': int(num_issues), 'precision': precision, 'recall': recall}

    # 3. TEMİZLEME STRATEJİLERİ
    print("\n[ADIM 3] Temizleme Stratejileri Karşılaştırılıyor...")
    
    # Strateji A: SİLME (DROP)
    print("  > Strateji: SİLME (DROP)")
    cleaned_dataset_drop = apply_drop_strategy(dataset, issues_mask)
    # Temizlenmiş verisetinde eğit
    # train_model fonksiyonunu dataset nesnesi alacak şekilde uyarlamak gerekir, ama şimdilik akışın çalıştığını kontrol ediyoruz
    # (Gerçek bir uygulamada, train_model'i dataset argümanı alacak şekilde refaktör ederdik)
    # ... Bu scripti basit tutmak için tam eğitimi atlıyoruz, ama mantık hazır.
    
    # Strateji B: YENİDEN ETİKETLEME (RELABEL)
    print("  > Strateji: YENİDEN ETİKETLEME (RELABEL)")
    cleaned_dataset_relabel = apply_relabel_strategy(dataset, issues_mask, psx)
    
    # Strateji C: YENİDEN AĞIRLIKLANDIRMA (REWEIGHT) (Bu script mantığında henüz tam uygulanmadı)
    
    print("İşlem hattı kontrolü tamamlandı. Stratejilerin gerçek eğitimi burada yapılacaktır.")
    
    # Sonuçları Kaydet
    with open(f'pipeline_results_{noise_type}_{noise_rate}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Sonuçlar kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', type=str, default='symmetric')
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--quick', action='store_true', help="Hata ayıklama için hızlı modda çalıştır (daha az epoch)")
    args = parser.parse_args()
    
    run_pipeline(args.noise_type, args.noise_rate, args.quick)
