import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


def _package_generic(logits, stage_maps, pooled_final, return_embeddings, embedding_layer):
    if embedding_layer is not None:
        if embedding_layer == 'final':
            return logits, pooled_final
        fmap = stage_maps[embedding_layer]
        return logits, torch.flatten(F.adaptive_avg_pool2d(fmap, 1), 1)

    if not return_embeddings:
        return logits

    embeddings = {name: torch.flatten(F.adaptive_avg_pool2d(fmap, 1), 1)
                  for name, fmap in stage_maps.items()}
    embeddings['final'] = pooled_final
    return logits, embeddings


# CIFAR ResNet (He et al. 2016) — the 6n+2 family: ResNet-20/32/44/56/110.

class CifarBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, depth: int = 20, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2 (e.g. 20, 32, 44, 56, 110)"
        n = (depth - 2) // 6

        self.in_planes = 16
        self.conv1 = nn.Conv2d(input_channels, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, n, stride=1)
        self.stage2 = self._make_stage(32, n, stride=2)
        self.stage3 = self._make_stage(64, n, stride=2)
        self.feat_dim = 64
        self.linear = nn.Linear(64, num_classes)

    def _make_stage(self, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(CifarBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_embeddings=False, embedding_layer=None):
        out = F.relu(self.bn1(self.conv1(x)))
        s1 = self.stage1(out)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        pooled_final = torch.flatten(F.adaptive_avg_pool2d(s3, 1), 1)
        logits = self.linear(pooled_final)

        stage_maps = {'stage_minus_1': s2, 'stage_minus_2': s1}
        return _package_generic(logits, stage_maps, pooled_final, return_embeddings, embedding_layer)


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen_factor: int = 10, num_classes: int = 10,
                 input_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4 (e.g. 16, 22, 28, 34, 40)"
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(input_channels, widths[0], 3, 1, 1, bias=False)
        self.stage1 = self._make_stage(widths[1], n, stride=1, dropout_rate=dropout_rate)
        self.stage2 = self._make_stage(widths[2], n, stride=2, dropout_rate=dropout_rate)
        self.stage3 = self._make_stage(widths[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn_final = nn.BatchNorm2d(widths[3])
        self.feat_dim = widths[3]
        self.linear = nn.Linear(widths[3], num_classes)

    def _make_stage(self, planes, n, stride, dropout_rate):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(WideBasicBlock(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_embeddings=False, embedding_layer=None):
        out = self.conv1(x)
        s1 = self.stage1(out)
        s2 = self.stage2(s1)
        s3_raw = self.stage3(s2)
        s3 = F.relu(self.bn_final(s3_raw))
        pooled_final = torch.flatten(F.adaptive_avg_pool2d(s3, 1), 1)
        logits = self.linear(pooled_final)

        stage_maps = {'stage_minus_1': s2, 'stage_minus_2': s1}
        return _package_generic(logits, stage_maps, pooled_final, return_embeddings, embedding_layer)


_TORCHVISION_RESNETS = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50}


class TorchvisionResNet(nn.Module):
    def __init__(self, arch: str = 'resnet50', num_classes: int = 10, input_channels: int = 3,
                 small_inputs: bool = True, pretrained: bool = False):
        super().__init__()
        backbone = _TORCHVISION_RESNETS[arch](weights='DEFAULT' if pretrained else None)

        if small_inputs:


            backbone.conv1 = nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()
        elif input_channels != 3:
            backbone.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4
        self.feat_dim = backbone.fc.in_features
        self.linear = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_embeddings=False, embedding_layer=None):
        out = self.stem(x)
        s1 = self.stage1(out)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        pooled_final = torch.flatten(F.adaptive_avg_pool2d(s4, 1), 1)
        logits = self.linear(pooled_final)

        stage_maps = {'stage_minus_1': s3, 'stage_minus_2': s2, 'stage_minus_3': s1}
        return _package_generic(logits, stage_maps, pooled_final, return_embeddings, embedding_layer)


def create_model(arch: str, num_classes: int, input_channels: int = 3,
                  small_inputs: bool = True, dropout_rate: float = 0.3,
                  pretrained: bool = False, **kwargs) -> nn.Module:
    """
    arch options:
      'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110'  
      'wrn-28-10', 'wrn-16-8', 'wrn-22-8',                   
      'pretrained_vit_b16'
      'resnet18', 'resnet34', 'resnet50'                           
      'mlp', 'frozen_enc_mlp'                                     

    small_inputs: True for CIFAR (32px) / Tiny-ImageNet (64px); False for full
                  ImageNet (224px). Only affects TorchvisionResNet's stem.
    **kwargs: extra model_params consumed by non-ResNet backbones (e.g. MLP's in_dim,
              hidden_dims, embed_dim); ignored by the ResNet families.
    """
    if arch.lower() == 'pretrained_vit_b16':
        from models.pretrained_vit import PretrainedViTB16
        return PretrainedViTB16(num_classes, **kwargs)
    arch = arch.lower()

    if arch in ('mlp', 'frozen_enc_mlp', 'tabular_mlp'):
        from models.mlp_architectures import create_mlp_model
        return create_mlp_model(arch, num_classes=num_classes, input_channels=input_channels, **kwargs)

    if arch == 'gpcnn':
        from models.text_architectures import create_text_model
        return create_text_model(arch, num_classes=num_classes, **kwargs)

    if arch.startswith('resnet') and arch not in _TORCHVISION_RESNETS and arch[6:].isdigit():
        depth = int(arch[6:])
        return CifarResNet(depth=depth, num_classes=num_classes, input_channels=input_channels)

    if arch.startswith('wrn-'):
        _, depth_str, widen_str = arch.split('-')
        return WideResNet(depth=int(depth_str), widen_factor=int(widen_str), num_classes=num_classes,
                           input_channels=input_channels, dropout_rate=dropout_rate)

    if arch in _TORCHVISION_RESNETS:
        return TorchvisionResNet(arch=arch, num_classes=num_classes, input_channels=input_channels,
                                  small_inputs=small_inputs, pretrained=pretrained)

    raise ValueError(f"Unrecognized arch: '{arch}'")
