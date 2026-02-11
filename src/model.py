import torch
import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=10, pretrained=False):
    """
    CIFAR-10 için uyarlanmış bir ResNet-18 modeli döndürür.
    """
    # Standart ResNet-18 kullanıyoruz ancak ilk evrişim (conv) katmanını değiştiriyoruz
    # çünkü CIFAR-10 görüntüleri 32x32 boyutunda, ImageNet gibi 224x224 değil.
    # Standart ResNet: conv1 7x7 stride 2 (agresif örnekleme azaltma/downsampling)
    # CIFAR ResNet: conv1 3x3 stride 1 (mekansal boyutları koru)
    
    model = models.resnet18(pretrained=pretrained)
    
    # 32x32 girdiyi daha iyi işlemek için ilk evrişim katmanını değiştir
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # maxpool katmanını kaldır (CIFAR-10 için yaygın bir uygulama, boyutu korumak için)
    model.maxpool = nn.Identity()
    
    # Son tam bağlantılı (fully connected) katmanı değiştir
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
