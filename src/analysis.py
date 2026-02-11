import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader
from src.dataset import CIFAR10Noise, get_transforms
from src.model import get_resnet18

def extract_features_and_loss(model, dataset, device='cuda'):
    """
    Modelden özellik vektörlerini (son katman öncesi) ve her örnek için kayıp değerini çıkarır.
    """
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    model.to(device)
    
    # Feature extractor: Son katmanı (fc) kaldırarak özellik vektörünü alacağız
    # ResNet18'de avgpool sonrası [Batch, 512, 1, 1] gelir, bunu düzleştireceğiz.
    # Ancak hooks kullanmak daha temiz olabilir veya fc katmanını Identity yapabiliriz kopyada.
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) # FC hariç her şey
    
    criterion = nn.CrossEntropyLoss(reduction='none') # Her örnek için ayrı kayıp
    
    features_list = []
    losses_list = []
    targets_list = []
    clean_targets_list = []
    
    with torch.no_grad():
        for inputs, targets, clean_targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Özellikler
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1) # Flatten
            features_list.append(features.cpu().numpy())
            
            # Kayıplar (Orijinal model ile tam ileri besleme lazım)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses_list.append(loss.cpu().numpy())
            
            targets_list.append(targets.cpu().numpy())
            clean_targets_list.append(clean_targets.numpy())
            
    return (np.concatenate(features_list), 
            np.concatenate(losses_list), 
            np.concatenate(targets_list), 
            np.concatenate(clean_targets_list))

def plot_tsne(features, targets, clean_targets, title="t-SNE Analizi", save_path="tsne.png"):
    """
    Özellik uzayını t-SNE ile 2D'ye indirger ve görselleştirir.
    Gürültülü örnekleri (targets != clean_targets) farklı işaretler.
    Hız için maksimum 2000 örnek kullanır.
    """
    print("t-SNE hesaplanıyor... (Bu işlem biraz zaman alabilir)")
    
    n_samples = min(2000, len(features))
    indices = np.random.choice(len(features), n_samples, replace=False)
    
    features_subset = features[indices]
    targets_subset = targets[indices]
    clean_targets_subset = clean_targets[indices]
    
    # Gürültü durumu: True ise etiket yanlıştır
    is_noisy = (targets_subset != clean_targets_subset)
    
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(features_subset)
    
    plt.figure(figsize=(10, 8))
    
    # Temiz örnekleri çiz
    plt.scatter(projections[~is_noisy, 0], projections[~is_noisy, 1], 
                c='blue', label='Temiz', alpha=0.5, s=10)
    
    # Gürültülü örnekleri çiz (Daha belirgin)
    plt.scatter(projections[is_noisy, 0], projections[is_noisy, 1], 
                c='red', label='Gürültülü (Etiket Hatası)', alpha=0.9, s=30, marker='x')
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"t-SNE grafiği kaydedildi: {save_path}")

def plot_loss_distribution(losses, targets, clean_targets, save_path="loss_dist.png"):
    """
    Temiz ve Gürültülü örneklerin kayıp dağılımlarını çizer.
    """
    is_noisy = (targets != clean_targets)
    
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(losses[~is_noisy], fill=True, color="blue", label="Temiz Örnekler", alpha=0.3)
    sns.kdeplot(losses[is_noisy], fill=True, color="red", label="Gürültülü Örnekler", alpha=0.3)
    
    plt.title("Kayıp Değerleri Dağılımı (Loss Distribution)")
    plt.xlabel("Kayıp (Loss)")
    plt.ylabel("Yoğunluk")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Kayıp dağılımı grafiği kaydedildi: {save_path}")
