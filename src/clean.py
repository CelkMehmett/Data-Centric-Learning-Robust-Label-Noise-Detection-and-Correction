import numpy as np
from cleanlab.filter import find_label_issues
from cleanlab.count import compute_confident_joint
from cleanlab.rank import get_label_quality_scores
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import CIFAR10Noise, get_transforms
from src.model import get_resnet18
import copy

def get_out_of_sample_probs(dataset, k=4, epochs=5, device='cuda'):
    """
    K-Fold CV kullanarak tüm veriseti için örnek dışı (out-of-sample) tahmin olasılıklarını hesaplar.
    Hız için epoch sayısı düşük tutulabilir.
    """
    num_samples = len(dataset)
    num_classes = len(dataset.classes)
    
    # Tüm olasılıklar için yer tutucu
    psx = np.zeros((num_samples, num_classes))
    
    # Katmanlar arasında sınıf dengesini korumak için Stratified K-Fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    s_targets = dataset.targets # Gürültülü etiketler
    
    print(f"Örnek dışı olasılıkları hesaplamak için {k}-Katlı Çapraz Doğrulama (CV) başlatılıyor...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(num_samples), s_targets)):
        print(f"  Kat (Fold) {fold+1}/{k}")
        
        # Alt kümeler (Subsets) oluştur
        train_sub = Subset(dataset, train_idx)
        val_sub = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=128, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_sub, batch_size=128, shuffle=False, num_workers=2)
        
        # Bu kat için modeli sıfırdan eğit
        model = get_resnet18(num_classes=10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for inputs, targets, _, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Doğrulama (Val) Katında Tahmin Yap
        model.eval()
        probs_fold = []
        with torch.no_grad():
            for inputs, _, _, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probs_fold.append(probs.cpu().numpy())
        
        psx[val_idx] = np.concatenate(probs_fold, axis=0)
        
    return psx

def detect_noise(dataset, psx):
    """
    Örnek dışı olasılıkları (psx) ve verilen gürültülü etiketleri kullanarak
    cleanlab ile etiket sorunlarını tespit eder.
    """
    noisy_labels = np.array(dataset.targets)
    
    # 1. Etiket Sorunlarını Bul (Boolean maskesi)
    # filter_by='prune_by_noise_rate' sağlam bir yöntemdir
    issues_mask = find_label_issues(
        labels=noisy_labels,
        pred_probs=psx,
        return_indices_ranked_by='self_confidence',
    )
    
    # 2. "Etiket kalitesine" göre sıralanmış örnekleri al (görselleştirme için)
    # Düşük skor = muhtemelen yanlış etiket
    quality_scores = get_label_quality_scores(
        labels=noisy_labels,
        pred_probs=psx
    )
    
    return issues_mask, quality_scores
