import torch
from src.dataset import CIFAR10Noise, get_transforms
from src.model import get_resnet18
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import os

def smoke_test():
    print("Duman Testi (Smoke Test) Çalıştırılıyor...")
    # Çok küçük bir alt küme kullan
    dataset = CIFAR10Noise(root='./data', train=True, download=False, transform=get_transforms(train=True))
    subset_indices = range(100) # Sadece 100 örnek
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=10)
    
    model = get_resnet18(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 1 epoch çalıştır
    model.train()
    for inputs, targets, _, _ in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print("Duman Testi Başarılı: Eğitim döngüsü hatasız çalışıyor.")

if __name__ == "__main__":
    smoke_test()
