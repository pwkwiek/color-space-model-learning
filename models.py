# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0

from colorspace import convert_and_normalize

# ------------------------------------------------------------
# ResNet18 (ImageNet-pretrained) with hooks
# ------------------------------------------------------------
class ResNet18_Pretrained(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

        self.first_layer_act = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.first_layer_act = x.detach().cpu()

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_first_conv(self):
        return self.conv1

    def get_last_conv_layer(self):
        return self.layer4[-1].conv2

    def extract_embeddings(self, x, color_space="rgb"):
        x = convert_and_normalize(x, color_space)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return x


# ------------------------------------------------------------
# MediumCNN / ShallowCNN / EfficientNetWrapper
# ------------------------------------------------------------
class MediumCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten_dim = None
        self.fc1 = None
        self.bn_fc = None
        self.fc2 = nn.Linear(512, num_classes)

        self.first_layer_act = None

    def _init_fc(self, x):
        self.flatten_dim = x.size(1)
        device = x.device
        self.fc1 = nn.Linear(self.flatten_dim, 512).to(device)
        self.bn_fc = nn.BatchNorm1d(512).to(device)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        self.first_layer_act = torch.nan_to_num(x.detach().cpu(), nan=0.0)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)

        if self.flatten_dim is None:
            self._init_fc(x)

        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_first_conv(self):
        return self.conv1

    def get_last_conv_layer(self):
        return self.conv4

    def extract_embeddings(self, x, color_space="rgb"):
        x = convert_and_normalize(x, color_space)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return x


class ShallowCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, first_layer_filters=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, first_layer_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(first_layer_filters)
        self.conv2 = nn.Conv2d(first_layer_filters, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)
        self.first_layer_act = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        self.first_layer_act = torch.nan_to_num(x.detach().cpu(), nan=0.0)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_first_conv(self):
        return self.conv1

    def get_last_conv_layer(self):
        return self.conv2

    def extract_embeddings(self, x, color_space="rgb"):
        x = convert_and_normalize(x, color_space)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return x


class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = efficientnet_b0(weights=None)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(base.classifier[1].in_features, num_classes)
        )
        self.first_layer_act = None
        self._first_conv = self.features[0][0]

    def forward(self, x):
        x_first = self._first_conv(x)
        self.first_layer_act = F.relu(x_first.detach().cpu())
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_first_conv(self):
        return self._first_conv

    def get_last_conv_layer(self):
        for m in reversed(list(self.features.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d layer in EfficientNet features.")

    def extract_embeddings(self, x, color_space="rgb"):
        x = convert_and_normalize(x, color_space)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# ------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        loss = logits[torch.arange(x.size(0)), class_idx].sum()
        loss.backward()

        grads = self.gradients
        activations = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


# ------------------------------------------------------------
# Model factory
# ------------------------------------------------------------
def get_model(model_name, num_classes):
    model_name = model_name.lower()
    if model_name == "resnet18":
        return ResNet18_Pretrained(num_classes=num_classes)
    elif model_name == "mediumcnn":
        return MediumCNN(in_channels=3, num_classes=num_classes)
    elif model_name == "shallowcnn":
        return ShallowCNN(in_channels=3, num_classes=num_classes, first_layer_filters=16)
    elif model_name == "efficientnet":
        return EfficientNetWrapper(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
