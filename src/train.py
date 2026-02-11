import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CIFAR10Noise, get_transforms
from src.model import get_resnet18
import time
import copy
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import os

def train_model(noise_type, noise_rate, num_epochs=20, batch_size=128, learning_rate=0.01, device='cuda'):
    print(f"Eğitim Başlatılıyor -> Gürültü Tipi: {noise_type}, Oran: {noise_rate}")
    
    # Cihazı ayarla
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # Verisetleri
    train_dataset = CIFAR10Noise(root='./data', train=True, download=True,
                                 transform=get_transforms(train=True),
                                 noise_type=noise_type, noise_rate=noise_rate)
    
    # Doğrulama (Validation) için, gerçek performansı ölçmek adına TEMİZ test setini kullanıyoruz
    test_dataset = CIFAR10Noise(root='./data', train=False, download=True,
                                transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = get_resnet18(num_classes=10).to(device)
    
    # Kayıp (Loss) ve Optimizasyon
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets, _, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        
        # Doğrulama (Validation)
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets, _, _ in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        print(f"Epoch {epoch+1}/{num_epochs} | Eğitim Kayıp: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

    # En iyi model ağırlıklarını yükle
    model.load_state_dict(best_model_wts)
    
    # Klasör yoksa oluştur
    os.makedirs('./models', exist_ok=True)
    save_path = f"./models/resnet18_{noise_type}_{noise_rate}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model kaydedildi: {save_path}")

    # Final Metrikler
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"En İyi Val Acc: {best_acc:.4f}")
    print(f"F1 Skoru (Macro): {f1:.4f}")
    
    return best_acc, f1, cm

if __name__ == "__main__":
    # Örnek kullanım: Temiz veri üzerinde eğit (%0 gürültü)
    train_model(noise_type=None, noise_rate=0.0, num_epochs=10)
