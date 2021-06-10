import timm
import torch
import torch.nn.functional as F

model = timm.create_model('tf_efficientnet_b4_ns', pretrained=False, num_classes=4)

x = torch.randn(2, 3, 224, 224)
y = model.forward_features(x)
y = F.adaptive_avg_pool2d(y, 1)

print(y.shape)
