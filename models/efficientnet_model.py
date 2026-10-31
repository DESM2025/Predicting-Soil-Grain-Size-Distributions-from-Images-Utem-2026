import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetGranulometria(nn.Module):
    def __init__(
        self,
        output_features: int = 11,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        if unfreeze_last_blocks < 0:
            raise ValueError("unfreeze_last_blocks no puede ser negativo.")

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        input_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(
            input_features,
            output_features,
        )

        if freeze_backbone:
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False
            if unfreeze_last_blocks > len(self.backbone.features):
                raise ValueError(
                    "unfreeze_last_blocks no puede superar la cantidad de bloques "
                    f"({len(self.backbone.features)})."
                )
            if unfreeze_last_blocks:
                for block in self.backbone.features[-unfreeze_last_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone(images)
