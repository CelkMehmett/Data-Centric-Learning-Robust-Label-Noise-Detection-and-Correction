import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import copy

class CIFAR10Noise(Dataset):
    def __init__(self, root, train=True, transform=None, download=True,
                 noise_type=None, noise_rate=0.0, random_seed=42):
        """
        Argümanlar:
            root (string): Verisetinin kök dizini.
            train (bool, opsiyonel): True ise eğitim setinden, aksi takdirde test setinden oluşturur.
            transform (callable, opsiyonel): PIL görüntüsünü alıp dönüştürülmüş versiyonunu döndüren fonksiyon/dönüşüm.
            download (bool, opsiyonel): True ise, verisetini internetten indirir ve root dizinine koyar.
            noise_type (string, opsiyonel): 'symmetric', 'asymmetric' (sınıf-bağımlı) veya None.
            noise_rate (float, opsiyonel): Gürültü oranı (0.0 ile 1.0 arası).
            random_seed (int, opsiyonel): Tekrarlanabilirlik için çekirdek (seed).
        """
        self.root = root
        self.transform = transform
        self.train = train
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_seed = random_seed

        # Orijinal CIFAR-10'u yükle
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=train, transform=None, download=download)
        self.data = self.cifar10.data
        self.targets = np.array(self.cifar10.targets)
        self.clean_targets = np.array(self.cifar10.targets) # Gerçek etiketler (Ground truth)
        self.classes = self.cifar10.classes
        
        # Gürültü enjeksiyonu
        if self.train and self.noise_type is not None and self.noise_rate > 0:
            self._inject_noise()

    def _inject_noise(self):
        np.random.seed(self.random_seed)
        num_samples = len(self.targets)
        num_classes = len(self.classes)
        
        # Bu değişken gürültülü etiketleri saklayacak
        new_targets = copy.deepcopy(self.targets)
        
        if self.noise_type == 'symmetric':
            # Simetrik gürültü: etiketi noise_rate olasılığıyla başka herhangi bir sınıfa çevir
            
            n_noisy = int(self.noise_rate * num_samples)
            noisy_indices = np.random.choice(num_samples, n_noisy, replace=False)
            
            for idx in noisy_indices:
                current_label = self.targets[idx]
                possible_labels = list(range(num_classes))
                possible_labels.remove(current_label)
                new_label = np.random.choice(possible_labels)
                new_targets[idx] = new_label
                
            print(f"Simetrik Gürültü Enjekte Ediliyor: %{self.noise_rate*100} -> {n_noisy} örnek çevrildi.")

        elif self.noise_type == 'asymmetric':
            # Sınıf-bağımlı gürültü (Benzer sınıfların karışmasını simüle eder)
            # CIFAR-10 için eşlemeler (makalelerde yaygın kullanılır):
            # Kamyon -> Otomobil, Kuş -> Uçak, Geyik -> At, Kedi -> Köpek
            # Kaynak: Patrini et al., 2017
            
            mapping = {
                9: 1, # Truck (Kamyon) -> Automobile (Otomobil)
                2: 0, # Bird (Kuş) -> Plane (Uçak)
                4: 7, # Deer (Geyik) -> Horse (At)
                3: 5, # Cat (Kedi) -> Dog (Köpek)
                5: 3, # Dog (Köpek) -> Cat (Kedi) (opsiyonel)
            }
            
            n_flipped = 0
            for i in range(num_samples):
                if self.targets[i] in mapping:
                    if np.random.random() < self.noise_rate:
                        new_targets[i] = mapping[self.targets[i]]
                        n_flipped += 1
            
            print(f"Asimetrik Gürültü Enjekte Ediliyor: Eşlenen sınıflarda %{self.noise_rate*100} oranında -> {n_flipped} örnek çevrildi.")

        self.targets = new_targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        clean_target = self.clean_targets[index]

        # PIL formatına dönüştür (torchvision dönüşümleri için gerekli)
        img =  torchvision.transforms.functional.to_pil_image(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, clean_target, index 

    def __len__(self):
        return len(self.data)

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
