import numpy as np
import copy
from src.dataset import CIFAR10Noise
from torch.utils.data import Subset, Dataset, WeightedRandomSampler
import torch
import torch.nn as nn

class CleanedDataset(Dataset):
    def __init__(self, original_dataset, keep_indices, new_labels=None):
        self.original_dataset = original_dataset
        self.keep_indices = keep_indices
        self.new_labels = new_labels # Orijinal indeksten yeni etikete harita (eğer yeniden etiketleme yapılıyorsa)

    def __getitem__(self, index):
        # indeksi orijinal indekse eşle
        original_idx = self.keep_indices[index]
        img, target, clean_target, _ = self.original_dataset[original_idx]
        
        # Eğer yeniden etiketliyorsak
        if self.new_labels is not None and original_idx in self.new_labels:
            target = self.new_labels[original_idx]
            
        return img, target, clean_target, original_idx

    def __len__(self):
        return len(self.keep_indices)

def apply_drop_strategy(dataset, issues_mask):
    """
    Etiket sorunu olduğu tespit edilen örnekleri kaldırır.
    """
    all_indices = np.arange(len(dataset))
    keep_indices = all_indices[~issues_mask]
    
    print(f"Silme (Drop) Stratejisi: {len(dataset)} örnekten {len(keep_indices)} tanesi tutuluyor.")
    return CleanedDataset(dataset, keep_indices)

def apply_relabel_strategy(dataset, issues_mask, pred_probs):
    """
    Sorunlu olduğu tespit edilen örnekleri modelin en yüksek olasılık verdiği sınıfla yeniden etiketler.
    """
    # Basitlik için tüm örnekleri tutuyoruz ama sorunlu olanların etiketini değiştiriyoruz
    all_indices = np.arange(len(dataset))
    
    # Yeni etiketleri belirle
    new_labels = {}
    issue_indices = np.where(issues_mask)[0]
    
    for idx in issue_indices:
        # Modelden en yüksek olasılığa sahip etiketi ata
        new_label = np.argmax(pred_probs[idx])
        new_labels[idx] = new_label
        
    print(f"Yeniden Etiketleme (Relabel) Stratejisi: {len(issue_indices)} örnek yeniden etiketleniyor.")
    # Tüm indeksleri geçiriyoruz ama bir etiket haritası (map) ile
    return CleanedDataset(dataset, all_indices, new_labels=new_labels)

def apply_reweight_strategy(dataset, issues_mask, confidence_scores):
    """
    Güvene dayalı özel bir kayıp fonksiyonu (veya ağırlıklar) döndürür.
    Bunu doğrudan standart CrossEntropy'ye entegre etmek zordur, reduction='none' kullanılmalı.
    Deneysel: WeightedRandomSampler için örnek ağırlıkları veya kayıp ağırlıkları döndürüyoruz.
    """
    # Basit yaklaşım: Ağırlık = Güven Skoru (Self-confidence)
    # Eğer güven düşükse, ağırlık da düşük olur.
    weights = confidence_scores
    
    # Ağırlıkları normalize et, böylece ortalama 1 olsun
    weights = weights / np.mean(weights)
    
    print(f"Yeniden Ağırlıklandırma (Reweight) Stratejisi: Ağırlıklar hesaplandı. Ort: {np.mean(weights):.4f}, Min: {np.min(weights):.4f}")
    return torch.FloatTensor(weights)
