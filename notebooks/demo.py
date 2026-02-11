import matplotlib.pyplot as plt
import numpy as np
from src.dataset import CIFAR10Noise, get_transforms
from src.clean import get_out_of_sample_probs
from cleanlab.filter import find_label_issues
import os
import torch

def show_samples(dataset, indices, title, filename):
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(indices[:10]): # İlk 10 tanesini göster
        img, target, clean_target, _ = dataset[idx]
        # Normalize işlemini geri al (Un-normalize)
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        img = img.clamp(0, 1)
        img_np = img.permute(1, 2, 0).numpy()
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img_np)
        plt.title(f"Verilen: {dataset.classes[target]}\nGerçek: {dataset.classes[clean_target]}", fontsize=8)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Kaydedildi: {filename}")

def run_visual_demo():
    print("Görsel Demo Oluşturuluyor...")
    # Veri yolu scriptin çalıştığı yere göre değişebilir, basitlik için üst dizini de kontrol edelim
    root_dir = '../data' if os.path.exists('../data') else './data'
    
    dataset = CIFAR10Noise(root=root_dir, train=True, download=True,
                           noise_type='symmetric', noise_rate=0.2,
                           transform=get_transforms(train=True))
    
    # 1. Rastgele Gürültülü Örnekleri Görselleştir
    noisy_indices = np.where(dataset.targets != dataset.clean_targets)[0]
    show_samples(dataset, noisy_indices, "Etiket Gürültüsü Örnekleri (Simetrik %20)", "noise_examples.png")
    
    # 2. Cleanlab Tespitlerini Görselleştir
    # (Demo hızı için küçük bir alt küme kullanılır veya zaten eğitilmiş modelden alınır)
    # Gerçek çalıştırmada: psx = get_out_of_sample_probs(dataset)
    print("Demo scripti tamamlandı. (Tam görselleştirme eğitilmiş model gerektirir)")

if __name__ == "__main__":
    run_visual_demo()
