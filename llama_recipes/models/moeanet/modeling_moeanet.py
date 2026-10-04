"""MoEA-Net model: visual experts, LLaMA fusion, and prognosis heads."""

import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from peft import prepare_model_for_kbit_training

from attention_expert import (
    SpatiotemporalExpert,
    SubretinalFluidClassification,
    VisionClassification,
)
from ..llama.modeling_llama import LlamaForCausalLM
from visual_encoder.biomedclip import load_biomedclip_visual_encoder


@dataclass
class PrognosisOutput:
    loss: torch.Tensor
    fluid_logits: torch.Tensor
    vision_logits: torch.Tensor


class ViTLlamaModel(nn.Module):
    """MoEA-Net prognosis model.

    The model keeps the frozen visual and language backbones separate from the
    trainable MoEA-Net components: modality experts, visual projector, special
    token embeddings, and prognosis heads.
    """

    vision_feature_dim = 768
    fluid_classes = 3
    vision_classes = 1
    default_checkpoint_name = "peft_model_lora_adapter"

    def __init__(self, train_config, use_cache, tokenizer, image_token_index, adapter_kwargs):
        super().__init__()
        self.train_config = train_config
        self.adapter_kwargs = adapter_kwargs
        self.training_stage = 1
        self.target_modalities = ["oct"]

        self.image_token_index = image_token_index
        self.fluid_token_index = tokenizer.convert_tokens_to_ids("<|fluid_change|>")
        self.vision_token_index = tokenizer.convert_tokens_to_ids("<|vision_change|>")
        self.special_token_ids = self._special_token_ids(tokenizer, image_token_index)

        self.language_model = self._build_language_model(train_config, use_cache, tokenizer)
        hidden_dim = self.language_model.config.hidden_size
        self._build_trainable_modules(hidden_dim)
        self._init_special_token_embeddings()
        self.vision_encoder = load_biomedclip_visual_encoder(
            getattr(train_config, "vision_encoder_path", None)
        )

    @staticmethod
    def _special_token_ids(tokenizer, image_token_index):
        return [
            token_id
            for token_id in tokenizer.additional_special_tokens_ids
            if token_id != image_token_index
        ]

    @staticmethod
    def _build_language_model(train_config, use_cache, tokenizer):
        language_model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_4bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )
        if len(tokenizer) > language_model.get_input_embeddings().num_embeddings:
            language_model.resize_token_embeddings(len(tokenizer))
        if train_config.quantization:
            language_model = prepare_model_for_kbit_training(language_model)
        return language_model

    def _build_trainable_modules(self, hidden_dim):
        # One SAE is assigned to each modality in the incremental schedule.
        self.experts = nn.ModuleList(
            SpatiotemporalExpert(
                emb=self.vision_feature_dim,
                temperature=getattr(self.train_config, "expert_temperature", 0.6),
            )
            for _ in range(3)
        )
        # Two-layer adaptor maps SAE features into the LLaMA embedding space.
        self.multi_modal_projector = nn.Sequential(
            nn.Linear(self.vision_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.subretinal_fluid_head = SubretinalFluidClassification(
            hidden_dim, self.fluid_classes
        )
        self.vision_head = VisionClassification(hidden_dim, self.vision_classes)

    def _init_special_token_embeddings(self):
        initial_tokens = self.language_model.get_input_embeddings().weight[
            self.special_token_ids
        ].detach().clone()
        self.special_token_embeddings = nn.Embedding.from_pretrained(
            initial_tokens, freeze=False
        )

    def move_to_device(self, device):
        """Move trainable modules while respecting automatic 4-bit LLM placement."""
        if not self.train_config.quantization:
            return self.to(device)
        for module in self._non_llm_modules():
            module.to(device)
        return self

    def _non_llm_modules(self):
        return (
            self.vision_encoder,
            self.experts,
            self.special_token_embeddings,
            self.multi_modal_projector,
            self.subretinal_fluid_head,
            self.vision_head,
        )

    def _extract_patch_tokens(self, images):
        activations = []

        def capture(_module, _inputs, output):
            activations.append(output)

        self.vision_encoder.eval()
        # Capture final patch tokens without changing the frozen BiomedCLIP forward API.
        hook = self.vision_encoder.trunk.blocks[-1].register_forward_hook(capture)
        try:
            with torch.no_grad():
                self.vision_encoder(images)
        finally:
            hook.remove()
        if len(activations) != 1 or activations[0].ndim != 3:
            raise RuntimeError("BiomedCLIP did not return patch tokens from its final block.")
        return activations[0][:, 1:, :]

    def _visual_features(self, modality_image_pairs):
        batch, modalities, images = self._validate_image_pairs(modality_image_pairs)
        patches = self._encode_images(images, batch, modalities)
        return self._apply_modality_experts(patches, modalities)

    def _validate_image_pairs(self, modality_image_pairs):
        if modality_image_pairs.ndim != 6 or modality_image_pairs.shape[2] != 2:
            raise ValueError("Expected image pairs with shape [B, M, 2, C, H, W].")
        batch, modalities, _, channels, height, width = modality_image_pairs.shape
        if modalities != self.training_stage:
            raise ValueError("The number of image modalities must match training_stage.")
        images = modality_image_pairs.reshape(-1, channels, height, width)
        return batch, modalities, images

    def _encode_images(self, images, batch, modalities):
        images = images.to(self._vision_device())
        patches = self._extract_patch_tokens(images)
        return patches.reshape(batch, modalities, 2, patches.shape[1], patches.shape[2])

    def _vision_device(self):
        return next(self.vision_encoder.parameters()).device

    def _apply_modality_experts(self, patches, modalities):
        features = []
        attention_maps = []
        # Modality order matches STAGES, e.g. OCT -> expert 0, fundus -> expert 1.
        for index in range(modalities):
            modality_features, before, after, patch_weights = self.experts[index](
                patches[:, index:index + 1]
            )
            features.append(modality_features)
            attention_maps.append((before, after, patch_weights))
        return torch.cat(features, dim=1), attention_maps

    @staticmethod
    def _token_positions(input_ids, token_id, token_name):
        matches = input_ids.eq(token_id)
        if not torch.all(matches.sum(dim=1) == 1):
            raise ValueError(f"Each prompt must contain exactly one {token_name} token.")
        return matches.long().argmax(dim=1)

    def _inject_visual_features(self, input_ids, visual_features):
        embeddings = self.language_model.get_input_embeddings()(input_ids).clone()
        self._replace_special_token_embeddings(input_ids, embeddings)

        # Projected visual tokens replace the prompt placeholders one-for-one.
        projected = self.multi_modal_projector(visual_features)
        projected = projected.reshape(projected.shape[0], -1, projected.shape[-1])
        image_positions = input_ids.eq(self.image_token_index)
        if not torch.all(image_positions.sum(dim=1) == projected.shape[1]):
            raise ValueError("Each image placeholder must match one projected visual token.")

        embeddings[image_positions] = projected.to(
            device=embeddings.device, dtype=embeddings.dtype
        ).reshape(-1, embeddings.shape[-1])
        return embeddings

    def _replace_special_token_embeddings(self, input_ids, embeddings):
        for index, token_id in enumerate(self.special_token_ids):
            positions = input_ids.eq(token_id)
            if positions.any():
                embeddings[positions] = self.special_token_embeddings.weight[index].to(
                    device=embeddings.device, dtype=embeddings.dtype
                )

    def _predict_logits(self, input_ids, attention_mask, modality_image_pairs):
        visual_features, attention_maps = self._visual_features(modality_image_pairs)
        input_ids, attention_mask = self._prepare_language_inputs(input_ids, attention_mask)
        visual_features = visual_features.to(self.multi_modal_projector[0].weight.device)
        embeddings = self._inject_visual_features(input_ids, visual_features)
        hidden = self._run_language_model(embeddings, attention_mask)
        fluid_hidden, vision_hidden = self._classification_hidden_states(input_ids, hidden)
        return (*self._classification_logits(fluid_hidden, vision_hidden), attention_maps)

    def _prepare_language_inputs(self, input_ids, attention_mask):
        embedding_device = self.language_model.get_input_embeddings().weight.device
        input_ids = input_ids.to(embedding_device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(embedding_device)
        return input_ids, attention_mask

    def _run_language_model(self, embeddings, attention_mask):
        outputs = self.language_model.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def _classification_hidden_states(self, input_ids, hidden):
        rows = torch.arange(input_ids.shape[0], device=hidden.device)
        fluid_positions = self._token_positions(
            input_ids, self.fluid_token_index, "fluid classification"
        )
        vision_positions = self._token_positions(
            input_ids, self.vision_token_index, "vision classification"
        )
        return (
            hidden[rows, fluid_positions.to(hidden.device)],
            hidden[rows, vision_positions.to(hidden.device)],
        )

    def _classification_logits(self, fluid_hidden, vision_hidden):
        fluid_weight = self.subretinal_fluid_head.fc.weight
        vision_weight = self.vision_head.fc.weight
        fluid_logits = self.subretinal_fluid_head(
            fluid_hidden.to(device=fluid_weight.device, dtype=fluid_weight.dtype)
        )
        vision_logits = self.vision_head(
            vision_hidden.to(device=vision_weight.device, dtype=vision_weight.dtype)
        )
        return fluid_logits, vision_logits

    def forward(self, input_ids, modality_image_pairs, labels, attention_mask=None, sample_id=None):
        fluid_logits, vision_logits, _ = self._predict_logits(
            input_ids, attention_mask, modality_image_pairs
        )
        fluid_labels = labels[:, 0].to(fluid_logits.device, dtype=torch.long)
        vision_labels = labels[:, 1].to(vision_logits.device, dtype=vision_logits.dtype)
        fluid_loss = F.cross_entropy(fluid_logits, fluid_labels)
        vision_loss = F.binary_cross_entropy_with_logits(
            vision_logits.squeeze(-1), vision_labels
        )
        loss = (
            self.train_config.fluid_loss_weight * fluid_loss
            + self.train_config.vision_loss_weight * vision_loss
        )
        return PrognosisOutput(loss, fluid_logits, vision_logits)

    @torch.no_grad()
    def generate(self, input_ids, modality_image_pairs, attention_mask=None, **_unused):
        fluid_logits, vision_logits, attention_maps = self._predict_logits(
            input_ids, attention_mask, modality_image_pairs
        )
        fluid_probabilities = F.softmax(fluid_logits, dim=-1)
        vision_probabilities = torch.sigmoid(vision_logits)
        return (
            fluid_probabilities.argmax(dim=-1),
            (vision_probabilities.squeeze(-1) > 0.5).long(),
            fluid_probabilities,
            vision_probabilities,
            attention_maps,
        )

    @staticmethod
    def _checkpoint_path(directory, name, stage, modalities, epoch):
        stem = f"{name}_{'_'.join(modalities)}_phase_{stage}"
        if epoch is not None:
            stem += f"_epoch_{epoch}"
        return os.path.join(directory, stem + ".pth")

    def save_pretrained(self, save_directory, peft_model_name=None, epoch=None):
        if len(self.target_modalities) != self.training_stage:
            raise ValueError("Checkpoint modality order must match training_stage.")
        peft_model_name = peft_model_name or self.default_checkpoint_name
        os.makedirs(save_directory, exist_ok=True)

        # Save only stage-owned components, shared heads/projector, and active LoRA branches.
        state = self._checkpoint_state(self.training_stage)
        active_adapters = self._active_adapter_names(self.training_stage)
        if not all(any(adapter in name for name in state) for adapter in active_adapters):
            raise RuntimeError("The active LoRA adapters are missing from the model state.")

        checkpoint = {
            "stage": self.training_stage,
            "modalities": list(self.target_modalities),
            "state_dict": state,
            "special_token_ids": self.special_token_ids,
            "expert_temperature": getattr(self.train_config, "expert_temperature", 0.6),
        }
        path = self._checkpoint_path(
            save_directory, peft_model_name, self.training_stage, self.target_modalities, epoch
        )
        torch.save(checkpoint, path)
        return path

    def from_pretrained(
        self,
        load_directory,
        peft_model_name=None,
        target_modalities=None,
        training_stage=1,
        epoch=None,
    ):
        peft_model_name = peft_model_name or self.default_checkpoint_name
        modalities = list(target_modalities or self.target_modalities)
        path = self._checkpoint_path(
            load_directory, peft_model_name, training_stage, modalities, epoch
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._validate_checkpoint_metadata(checkpoint, training_stage, modalities)

        # Recreate all adapters used up to this stage before loading weights.
        adapter_names = self._active_adapter_names(training_stage)
        self._ensure_adapters(adapter_names)
        self.language_model.set_adapter(adapter_names)

        state = checkpoint["state_dict"]
        expected = self._expected_checkpoint_keys(training_stage, adapter_names)
        if set(state) != expected:
            raise ValueError("Checkpoint is missing or contains unexpected stage parameters.")
        self.load_state_dict(state, strict=False)
        self.training_stage = training_stage
        self.target_modalities = modalities
        return path

    def _checkpoint_state(self, stage):
        expected = self._expected_checkpoint_keys(stage, self._active_adapter_names(stage))
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in expected
        }

    def _expected_checkpoint_keys(self, stage, adapter_names):
        active_experts = self._active_expert_prefixes(stage)
        shared_modules = self._shared_checkpoint_prefixes()
        return {
            name
            for name in self.state_dict()
            if name.startswith(active_experts + shared_modules)
            or (
                name.startswith("language_model.")
                and any(adapter in name for adapter in adapter_names)
            )
        }

    @staticmethod
    def _active_expert_prefixes(stage):
        return tuple(f"experts.{index}." for index in range(stage))

    @staticmethod
    def _shared_checkpoint_prefixes():
        return (
            "special_token_embeddings.",
            "multi_modal_projector.",
            "subretinal_fluid_head.",
            "vision_head.",
        )

    @staticmethod
    def _active_adapter_names(stage):
        return [f"lora_phase_{index}" for index in range(1, stage + 1)]

    def _validate_checkpoint_metadata(self, checkpoint, training_stage, modalities):
        if checkpoint.get("stage") != training_stage or checkpoint.get("modalities") != modalities:
            raise ValueError("Checkpoint stage or modality order does not match the request.")
        if checkpoint["special_token_ids"] != self.special_token_ids:
            raise ValueError("Checkpoint tokenizer special tokens do not match this model.")
        if "expert_temperature" in checkpoint:
            current_temperature = getattr(self.train_config, "expert_temperature", 0.6)
            if checkpoint["expert_temperature"] != current_temperature:
                raise ValueError("Checkpoint expert_temperature does not match this model.")

    def _ensure_adapters(self, adapter_names):
        existing = getattr(self.language_model, "peft_config", None) or {}
        missing = [name for name in adapter_names if name not in existing]
        if not missing:
            return

        from llama_recipes.utils.config_utils import generate_peft_config

        for name in missing:
            self.language_model.add_adapter(
                adapter_config=generate_peft_config(self.train_config, self.adapter_kwargs),
                adapter_name=name,
            )
