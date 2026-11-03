from .resnet1d import resnet1d_18, resnet1d_50
from .vit import ViTEncoder

ENCODERS = {"vit_tiny": ViTEncoder, "resnet1d_18": resnet1d_18, "resnet1d_50": resnet1d_50}


def build_encoder(name):
    return ENCODERS[name]()
