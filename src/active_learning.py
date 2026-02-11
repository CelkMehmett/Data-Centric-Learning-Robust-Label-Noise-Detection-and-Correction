import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.dataset import CIFAR10Noise, get_transforms
from src.model import get_resnet18
import copy

class ActiveLearner:
    def __init__(self, dataset, model, initial_labeled_size=1000, batch_size=64, device='cuda'):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # Etiketlenmemiş havuz indeksleri (başlangıçta hepsi)
        self.unlabeled_indices = np.arange(len(dataset))
        self.labeled_indices = []
        
        # Rastgele bir alt kümeyle başla
        self.query(initial_labeled_size, strategy='random')

    def train_model(self, epochs=5):
        """Modeli şu anki etiketli veriler üzerinde eğit"""
        subset = Subset(self.dataset, self.labeled_indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        self.model.train()
        for epoch in range(epochs):
            for inputs, targets, _, _ in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def get_uncertainties(self):
        """Etiketlenmemiş havuz için belirsizliği (marjı) hesapla"""
        subset = Subset(self.dataset, self.unlabeled_indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for inputs, _, _, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                # Marj Örneklemesi (Margin Sampling): 1 - (Enİyi - İkinciEnİyi)
                # Küçük marj = Yüksek belirsizlik
                top2 = torch.topk(probs, 2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                uncertainty = 1 - margin
                
                uncertainties.extend(uncertainty.cpu().numpy())
                
        return np.array(uncertainties)

    def query(self, n_samples, strategy='uncertainty'):
        if len(self.unlabeled_indices) == 0:
            return
            
        if strategy == 'random':
            indices_to_label = np.random.choice(self.unlabeled_indices, size=n_samples, replace=False)
        elif strategy == 'uncertainty':
            uncertainties = self.get_uncertainties()
            # EN YÜKSEK belirsizliğe sahip örnekleri seç
            # Azalan şekilde sırala
            top_k_idx = np.argsort(uncertainties)[-n_samples:]
            indices_to_label = self.unlabeled_indices[top_k_idx]
        
        # "Oracle" (Kahin) adımı: Bu indeksler için etiketleri düzelt
        # Simülasyonumuzda, dataset.targets'ı clean_targets ile güncelliyoruz
        for idx in indices_to_label:
            # Etiketi düzelt!
            self.dataset.targets[idx] = self.dataset.clean_targets[idx]
        
        # Havuzları güncelle
        self.labeled_indices.extend(indices_to_label)
        self.unlabeled_indices = np.setdiff1d(self.unlabeled_indices, indices_to_label)
        
        print(f"{strategy} stratejisi kullanılarak {n_samples} örnek sorgulandı. Toplam Etiketli (Düzeltilmiş): {len(self.labeled_indices)}")

def run_active_learning_loop(steps=5, samples_per_step=500):
    # Kurulum
    dataset = CIFAR10Noise(root='./data', train=True, download=True, noise_type='symmetric', noise_rate=0.4)
    model = get_resnet18(num_classes=10).cuda()
    
    learner = ActiveLearner(dataset, model)
    
    accuracies = []
    
    for i in range(steps):
        print(f"AL Adım {i+1}/{steps}")
        learner.train_model(epochs=3) # Mevcut set üzerinde eğit
        
        # Test seti üzerinde değerlendir (Temiz)
        test_dataset = CIFAR10Noise(root='./data', train=False)
        test_loader = DataLoader(test_dataset, batch_size=128)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets, _, _ in test_loader:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = correct / total
        accuracies.append(acc)
        print(f"  Test Doğruluğu (Acc): {acc:.4f}")
        
        # Daha fazla sorgula
        learner.query(samples_per_step, strategy='uncertainty')
        
    return accuracies

if __name__ == "__main__":
    run_active_learning_loop()
