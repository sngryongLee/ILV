import timm
import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange


class DinoWrapper(nn.Module):
    """
    Dino v1 wrapper using timm implementation.
    Extracts patch embeddings and returns features without class token.
    """

    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):

        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        x = self.processor(image)
        feats = self.model.forward_features(x)

        return feats[:, 1:]

    def freeze(self, is_train=False):
        print(f"======== Dino encoder is_train: {is_train} ========")
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        for _, param in self.model.named_parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries=3, proxy_error_cooldown=5):
        import requests
        try:
            ##############
            model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
            data_cfg = timm.data.resolve_model_data_config(model)
            processor = transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"])
            return model, processor

        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: retrying in {proxy_error_cooldown} seconds…")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(
                    model_name,
                    proxy_error_retries=proxy_error_retries - 1,
                    proxy_error_cooldown=proxy_error_cooldown,
                )
            raise err
