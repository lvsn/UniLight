from typing import Optional, Tuple, Union, Any
from dataclasses import dataclass
import random
import json
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

from transformers.utils import logging, ModelOutput
from transformers.models.auto.modeling_auto import AutoModel
from transformers import Dinov2Model, DINOv3ViTModel, Qwen3Model, AutoTokenizer

from difflib import get_close_matches
from light_utils import tokenize_light_descriptions, preprocess_image

from light_decoder import SHPredictionHead

logger = logging.get_logger(__name__)


def _convert_to_serializable(obj):
    """
    Convert DictConfig and other non-serializable objects to JSON-serializable format.
    """
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def dummy_loss(logits: torch.Tensor) -> torch.Tensor:
    return 0.0 * logits.sum()  # Dummy loss, to avoid unused parameters error in distributed training


def qwen_last_token_pool(last_hidden_states: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
    # left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    # if left_padding:
    #     return last_hidden_states[:, -1]
    # else:
    #     sequence_lengths = attention_mask.sum(dim=1) - 1
    #     batch_size = last_hidden_states.shape[0]
    #     return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    # We don't need to check for left padding, qwen is by default left padded
    return last_hidden_states[:, -1]


class AttentionFusionModule(nn.Module):
    def __init__(self, embed_dim: int, num_summary_tokens: int = 8, n_heads: int = 8, use_residual_pooled_input: bool = False):
        super().__init__()
        self.num_summary_tokens = num_summary_tokens
        self.use_residual_pooled_input = use_residual_pooled_input

        # Learnable tokens that will act as the "queries" to summarize the input
        self.summary_queries = nn.Parameter(torch.randn(1, num_summary_tokens, embed_dim))

        # Standard Multi-Head Attention layer
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_tokens: torch.Tensor):
        # input_tokens shape: [B, N, D] (from DINOv2 or Qwen2)
        batch_size = input_tokens.shape[0]

        # Expand summary queries to match the batch size
        queries = self.summary_queries.expand(batch_size, -1, -1)  # Shape: [B, K, D]

        # Use the summary tokens to query the input tokens
        # It learns to "pull" the most important info from the input sequence
        summary_output, _ = self.attention(
            query=queries,
            key=input_tokens,
            value=input_tokens
        )
        summary_output = self.norm(summary_output)  # Shape: [B, K, D]

        if self.use_residual_pooled_input:
            # Add residual connection from the input, which is adaptively pooled to the same number of tokens
            pooled_input = F.adaptive_avg_pool1d(input_tokens.transpose(1, 2), self.num_summary_tokens).transpose(1, 2)
            summary_output += pooled_input

        return summary_output


@dataclass
class LightMultiEncoderOutput(ModelOutput):
    multimodal_embeddings: Optional[dict] = None
    sh_predictions: Optional[dict] = None


class LightMultiEncoderModel(nn.Module):
    base_model_prefix = "light_multi_encoder"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: dict = None,
        encoder_dict: Optional[dict] = None,
        tokenizer_dict: Optional[dict] = None,
        projection_dim: int = 512,
        # Flag to enable the summary token mechanism
        use_summary_tokens: bool = True,
        num_summary_tokens: int = 8,
        # Optional logit scale for contrastive loss
        use_logit_scale: bool = False,
        logit_scale_init_value: float = 2.6592,
        # SH prediction head options
        use_sh_prediction: bool = True,
        sh_order: int = 3,
        sh_hidden_dims: list = [1024, 1024],
        **kwargs,
    ):
        super().__init__()

        # Store initialization kwargs for saving/loading
        self._init_kwargs = {
            'projection_dim': projection_dim,
            'use_summary_tokens': use_summary_tokens,
            'num_summary_tokens': num_summary_tokens,
            'use_logit_scale': use_logit_scale,
            'logit_scale_init_value': logit_scale_init_value,
            'use_sh_prediction': use_sh_prediction,
            'sh_order': sh_order,
            'sh_hidden_dims': sh_hidden_dims,
        }
        self._init_kwargs.update(kwargs)

        self.projection_dim = projection_dim
        self.light_modalities = list(encoder_dict.keys())
        self.encoder_dict = encoder_dict
        self.tokenizer_dict = tokenizer_dict

        # Store summary token configuration
        self.use_summary_tokens = use_summary_tokens
        self.num_summary_tokens = num_summary_tokens

        # Store SH prediction configuration
        self.use_sh_prediction = use_sh_prediction
        self.sh_order = sh_order
        self.n_sh = (sh_order + 1) ** 2
        self.sh_hidden_dims = sh_hidden_dims

        embed_dim_dict = {}
        self.projection_dict = nn.ModuleDict()

        if self.use_summary_tokens:
            self.fusion_dict = nn.ModuleDict()

        if self.use_sh_prediction:
            self.sh_head_dict = nn.ModuleDict()

        self.use_logit_scale = use_logit_scale
        self.logit_scale_dict = nn.ParameterDict()

        # Set up the encoders, projections, and optional fusion layers
        for i, encoder_name in enumerate(self.light_modalities):
            embed_dim = config[encoder_name].hidden_size
            embed_dim_dict[encoder_name] = embed_dim

            if self.use_summary_tokens:
                self.fusion_dict[encoder_name] = AttentionFusionModule(embed_dim=embed_dim, num_summary_tokens=num_summary_tokens, use_residual_pooled_input=False)
                # self.fusion_dict[encoder_name] = AttentionFusionModule(embed_dim=embed_dim, num_summary_tokens=num_summary_tokens, use_residual_pooled_input=True)
                self.projection_dict[encoder_name] = nn.Linear(embed_dim, projection_dim, bias=False)
            else:
                self.projection_dict[encoder_name] = nn.Linear(embed_dim, projection_dim, bias=False)

            # Add SH prediction head for all modalities when enabled
            if self.use_sh_prediction:
                self.sh_head_dict[encoder_name] = SHPredictionHead(
                    num_tokens=self.num_summary_tokens,
                    input_dim=projection_dim,
                    sh_order=self.sh_order,
                    hidden_dims=self.sh_hidden_dims
                )

            for j, encoder_name2 in enumerate(self.light_modalities):
                if i >= j:
                    continue
                if self.use_logit_scale:
                    self.logit_scale_dict[f"{encoder_name}_to_{encoder_name2}"] = nn.Parameter(torch.tensor(logit_scale_init_value))

    def get_modal_features(
        self,
        modal: str,
        modal_data: Union[torch.Tensor, Any],
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        modal_embeds = {}

        # 1. Find the correct encoder for the modality
        encoder = None
        closest_key = modal
        if modal not in self.encoder_dict:
            closest_key = get_close_matches(modal, self.encoder_dict.keys(), n=1, cutoff=0.1)[0]
            # logger.warning(f"Modal {modal} not found in encoder_dict. Using the first match: {closest_key}.")
        encoder = self.encoder_dict[closest_key]

        # 2. Get the full sequence output from the encoder
        if 'envmap' in modal or 'irradiance' in modal or 'rgb' in modal:
            if isinstance(modal_data, (str, Image.Image)):
                modal_data = preprocess_image(modal, modal_data).unsqueeze(0).to(next(encoder.parameters()).device)
            encoder_kwargs = {
                'pixel_values': modal_data,
                'output_attentions': output_attentions,
                'output_hidden_states': output_hidden_states,
                'return_dict': return_dict,
            }
            outputs = encoder(**encoder_kwargs)
            sequence_output = outputs.last_hidden_state
            pooled_output_original = outputs.pooler_output
        elif 'light_description' in modal:
            if isinstance(modal_data, list) or isinstance(modal_data, str):
                if isinstance(modal_data, str):
                    modal_data = [modal_data]
                # If modal_data is a list of strings, we directly tokenize them
                tokenizer = self.tokenizer_dict[closest_key]
                modal_data = tokenize_light_descriptions(
                    tokenizer=tokenizer,
                    light_descriptions=modal_data,
                    device=encoder.device
                )
            outputs = encoder(
                input_ids=modal_data.input_ids,
                attention_mask=modal_data.attention_mask,
                position_ids=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            sequence_output = outputs.last_hidden_state
            pooled_output_original = qwen_last_token_pool(last_hidden_states=sequence_output, attention_mask=modal_data.attention_mask)
        else:
            raise ValueError(f"Unsupported modal type: {modal}.")

        modal_embeds[modal + '_raw'] = sequence_output

        # 3. Process the sequence to get the final embedding
        if self.use_summary_tokens:
            # --- NEW PATH: Use AttentionFusionModule ---
            summary_tokens = self.fusion_dict[closest_key](sequence_output)
            projected_summary = self.projection_dict[closest_key](summary_tokens)  # Shape: [B, K, D]

            mu = projected_summary

        else:
            # --- ORIGINAL PATH: Use standard pooling ---
            mu = self.projection_dict[closest_key](pooled_output_original)

        # 4. Estimate the spherical harmonics parameters if enabled
        if self.use_sh_prediction:
            sh_pred = self.sh_head_dict[closest_key](mu)  # Shape: (B, 3 * N_sh)
            modal_embeds[modal + '_sh_pred'] = sh_pred

        # Common final steps for both paths
        modal_embeds[modal + '_mu'] = mu

        embeds = F.normalize(mu, p=2, dim=-1)
        modal_embeds[modal + '_embeds'] = embeds

        return modal_embeds

    def forward(
        self,
        multimodal_input_dict: Optional[dict] = {
            'envmap': None,
            'irradiance': None,
            'light_description': None,
        },
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], LightMultiEncoderOutput]:

        multimodal_embeds = {}

        for modal, modal_data in multimodal_input_dict.items():
            if modal_data is None:
                continue
            # get_modal_features now returns a dictionary, so we update
            multimodal_embeds.update(
                self.get_modal_features(
                    modal=modal,
                    modal_data=modal_data,
                    token_type_ids=token_type_ids,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            )

        output = multimodal_embeds
        return output if not return_dict else LightMultiEncoderOutput(
            logits_all=None,
            multimodal_embeddings=multimodal_embeds,
        )

    @classmethod
    def from_encoder_configs_init(
        cls,
        encoder_init_configs: dict,
        *model_args,
        **kwargs,
    ) -> "LightMultiEncoderModel":

        encoder_dict = {}
        encoder_config = {}

        tokenizer_dict = {}

        for encoder_name, encoder_init_config in encoder_init_configs.items():
            if 'model_name_or_path' not in encoder_init_config:
                raise ValueError(f"`{encoder_name}_model_name_or_path` can not be `None`.")

            model_class = AutoModel

            model_specific_kwargs = {k.replace(f"{encoder_name}_", ""): v for k, v in kwargs.items() if k.startswith(f"{encoder_name}_")}
            # Filter out kwargs that are for the LightMultiEncoderModel itself
            main_model_arg_names = cls.__init__.__code__.co_varnames
            model_specific_kwargs = {k: v for k, v in model_specific_kwargs.items() if k not in main_model_arg_names}

            model = model_class.from_pretrained(
                encoder_init_config['model_name_or_path'],
                **model_specific_kwargs
            )

            # Modify the input channels for vision models
            if change_input_channels := encoder_init_config.get('change_input_channels_to', None):
                if not isinstance(model, (Dinov2Model, DINOv3ViTModel)):
                    continue
                if isinstance(model, Dinov2Model):
                    old_proj = model.embeddings.patch_embeddings.projection
                elif isinstance(model, DINOv3ViTModel):
                    old_proj = model.embeddings.patch_embeddings
                new_proj = nn.Conv2d(change_input_channels, old_proj.out_channels,
                                     kernel_size=old_proj.kernel_size, stride=old_proj.stride,
                                     padding=old_proj.padding, bias=old_proj.bias is not None)

                # Init the new projection layer to all zeros
                nn.init.zeros_(new_proj.weight)
                if new_proj.bias is not None:
                    nn.init.zeros_(new_proj.bias)

                with torch.no_grad():
                    new_proj.weight[:, :min(3, change_input_channels), :, :].copy_(old_proj.weight[:, :3, :, :])
                    if old_proj.bias is not None:
                        new_proj.bias.copy_(old_proj.bias)

                if isinstance(model, Dinov2Model):
                    model.embeddings.patch_embeddings.projection = new_proj
                    model.config.num_channels = change_input_channels
                    model.embeddings.patch_embeddings.num_channels = change_input_channels
                elif isinstance(model, DINOv3ViTModel):
                    model.embeddings.patch_embeddings = new_proj
                    model.config.num_channels = change_input_channels
                    model.embeddings.config.num_channels = change_input_channels

            if isinstance(model, Qwen3Model):
                tokenizer = AutoTokenizer.from_pretrained(encoder_init_config['model_name_or_path'], use_fast=True, padding_side='left')
                tokenizer_dict[encoder_name] = tokenizer

            encoder_dict[encoder_name] = model

        encoder_dict = nn.ModuleDict(encoder_dict)

        for encoder_name in encoder_dict.keys():
            encoder_config[encoder_name] = encoder_dict[encoder_name].config

        model = cls(config=encoder_config, encoder_dict=encoder_dict, tokenizer_dict=tokenizer_dict, **kwargs)

        # Store encoder initialization configs for saving
        model._encoder_init_configs = encoder_init_configs

        # Store source modalities mapping if available
        model.source_modalities_mapping = {}
        for encoder_name, config in encoder_init_configs.items():
            if 'source_modalities' in config:
                model.source_modalities_mapping[encoder_name] = config['source_modalities']

        return model

    def freeze_embeddings_mask_token(self):
        for name, encoder in self.encoder_dict.items():
            if isinstance(encoder, Dinov2Model) or isinstance(encoder, DINOv3ViTModel):
                encoder.embeddings.mask_token.requires_grad_(False)

    def freeze_encoder_unfreeze_projection(self):
        self.requires_grad_(False)

        for name, encoder in self.encoder_dict.items():
            if 'envmap' in name or 'irradiance' in name:
                encoder.embeddings.patch_embeddings.requires_grad_(True)

        self.projection_dict.requires_grad_(True)
        if self.use_summary_tokens:
            self.fusion_dict.requires_grad_(True)
        if self.use_sh_prediction:
            self.sh_head_dict.requires_grad_(True)

    def freeze_vision_unfreeze_text(self):
        for name, encoder in self.encoder_dict.items():
            if 'envmap' in name or 'irradiance' in name or 'rgb' in name:
                encoder.requires_grad_(False)
            elif 'light_description' in name:
                encoder.requires_grad_(True)

        for name, encoder in self.encoder_dict.items():
            if 'envmap' in name or 'irradiance' in name:
                encoder.embeddings.patch_embeddings.requires_grad_(True)

        self.projection_dict.requires_grad_(True)
        if self.use_summary_tokens:
            self.fusion_dict.requires_grad_(True)
        if self.use_sh_prediction:
            self.sh_head_dict.requires_grad_(True)

    def unfreeze_all(self):
        self.requires_grad_(True)
        self.freeze_embeddings_mask_token()

    def get_backbone_params(self):
        """
        Returns the parameters of the backbone encoders.
        This is useful for optimizers that need to differentiate between backbone and projection parameters.
        """
        return [param for encoder in self.encoder_dict.values() for param in encoder.parameters()]

    def get_projection_params(self):
        """
        Returns the parameters of the projection layers.
        This is useful for optimizers that need to differentiate between backbone and projection parameters.
        """
        params = list(self.projection_dict.parameters())
        if self.use_summary_tokens:
            params.extend(list(self.fusion_dict.parameters()))
        return params

    def get_sh_head_params(self):
        """
        Returns the parameters of the SH prediction heads.
        This is useful for optimizers that need to differentiate between backbone and projection parameters.
        """
        if self.use_sh_prediction:
            return list(self.sh_head_dict.parameters())
        else:
            return None

    def freeze_encoder_unfreeze_sh_head(self):
        """
        Freeze all encoder and projection parameters, only unfreeze the SH prediction head.
        This is useful for linear probing experiments.
        """
        # Freeze everything first
        self.requires_grad_(False)

        # Unfreeze only the SH prediction heads
        if self.use_sh_prediction:
            self.sh_head_dict.requires_grad_(True)
            logger.info("Froze all encoders and projections, unfroze SH prediction heads")
        else:
            logger.warning("No SH prediction heads to unfreeze")

    def save_pretrained(self, save_directory: str):
        """
        Save the model and its configuration to a directory.

        Args:
            save_directory: Directory where to save the model.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save model weights
        model_path = os.path.join(save_directory, "model.safetensors")

        # Use safetensors if available, otherwise use torch.save
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), model_path)
        except ImportError:
            model_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(self.state_dict(), model_path)

        # Save configuration
        config = {
            'encoder_init_configs': _convert_to_serializable(getattr(self, '_encoder_init_configs', {})),
            'model_kwargs': _convert_to_serializable(self._init_kwargs),
            'light_modalities': self.light_modalities,
        }

        config_path = os.path.join(save_directory, "unilight_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "LightMultiEncoderModel":
        """
        Load a pretrained model from a directory.

        Args:
            model_path: Path to the directory containing the saved model.
            **kwargs: Additional keyword arguments to override saved configuration.

        Returns:
            LightMultiEncoderModel: The loaded model.
        """
        # Load configuration
        config_path = os.path.join(model_path, "unilight_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No unilight_config.json found in {model_path}")

        with open(config_path, 'r') as f:
            saved_config = json.load(f)

        # Merge saved config with provided kwargs
        encoder_init_configs = saved_config.get('encoder_init_configs', {})
        model_kwargs = saved_config.get('model_kwargs', {})
        model_kwargs.update(kwargs)  # Override with any provided kwargs

        # Initialize the model using the saved encoder configs
        if encoder_init_configs:
            model = cls.from_encoder_configs_init(
                encoder_init_configs=encoder_init_configs,
                **model_kwargs
            )
        else:
            raise ValueError("No encoder_init_configs found in saved configuration")

        # Load model weights
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
            except ImportError:
                raise ImportError("safetensors is required to load this model. Install it with: pip install safetensors")
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"No model weights found in {model_path}")

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys when loading model: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

        logger.info(f"Model loaded from {model_path}")
        return model


__all__ = ["LightMultiEncoderModel"]
