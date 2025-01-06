from typing import Optional, Tuple
import torch
import torch.nn as nn

class SigLipVisionConfig:

    def __init__(
            self,
            hidden_size = 768,
            intermidiate_size = 3072,
            num_layers = 12,
            num_attention_heads = 12,
            num_channels = 3,
            image_size = 224,
            patch_size = 16,
            layer_norm_eps = 1e-6,
            attention_dropout=0.0,
            num_image_tokens: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermidiate_size = intermidiate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SigLipVisionEmbeddings(nn.Module):

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.emb_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size
    
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.emb_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            paddind = "valid"
        )



class SigLipVisionTransformer(nn.Module):

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        emb_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(emb_dim, ps = config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor : 

        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SigLipVisionModel(nn.Module):

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)
        
    def forward(self, pixel_values) -> Tuple:
        """
        my vit would convert {batch,channels,h,w} -> {batch,num_patches,emb_dim}
        """
        return self.vision_model(pixel_values = pixel_values)