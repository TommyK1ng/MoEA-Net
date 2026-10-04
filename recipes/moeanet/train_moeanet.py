"""Train MoEA-Net with progressive modality stages."""

import math
import os
import random
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Allow running this file directly from recipes/moeanet without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.datasets import OctEyeDataset
from llama_recipes.models.moeanet.modeling_moeanet import ViTLlamaModel
from llama_recipes.prognosis import (
    STAGES,
    load_prognosis_tokenizer,
    stage_batch_size,
    stage_epoch_count,
    stage_output_dir,
)
from llama_recipes.utils.config_utils import generate_peft_config, update_config
from llama_recipes.utils.prognosis_training import fit_prognosis

# Tokenizer multiprocessing can conflict with PyTorch dataloader workers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda:0"



def validate_training_config(train_config):
    """Fail early on invalid loss weights or missing CUDA."""
    weights = (train_config.fluid_loss_weight, train_config.vision_loss_weight)
    expert_temperature = getattr(train_config, "expert_temperature", 0.6)
    if any(not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0
           for weight in weights) or not any(weights):
        raise ValueError("Loss weights must be finite, non-negative, and not both zero.")
    if (
        not isinstance(expert_temperature, (int, float))
        or not math.isfinite(expert_temperature)
        or expert_temperature <= 0
    ):
        raise ValueError("expert_temperature must be finite and positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("This 8B model requires a CUDA GPU for training.")


def seed_everything(seed):
    """Make stage comparisons repeatable across Python, NumPy, and PyTorch."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(train_config, modalities, tokenizer, split, batch_size):
    """Create the stage-specific dataset filtered by required modalities."""
    dataset = OctEyeDataset(
        train_config.dataset_directory, list(modalities), tokenizer, train_config, split
    )
    if not dataset:
        raise ValueError(f"No {split} samples contain all modalities: {modalities}")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
    )


def configure_stage(model, stage, modalities, peft_config):
    """Activate the current stage while freezing earlier LoRA and expert modules."""
    model.training_stage = stage
    model.target_modalities = list(modalities)

    adapter_name = f"lora_phase_{stage}"
    # All adapters remain active for inference, but only the newest one is trainable.
    model.language_model.add_adapter(adapter_config=peft_config, adapter_name=adapter_name)
    model.language_model.set_adapter([f"lora_phase_{index}" for index in range(1, stage + 1)])

    for name, parameter in model.language_model.named_parameters():
        parameter.requires_grad_(adapter_name in name)
    for index, expert in enumerate(model.experts):
        expert.requires_grad_(index == stage - 1)


def split_trainable_parameters(model):
    """Use separate learning rates for LoRA and non-LoRA trainable modules."""
    base_params = []
    lora_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = lora_params if "lora_phase_" in name else base_params
        target.append(parameter)
    if not lora_params or not base_params:
        raise RuntimeError("The current stage needs trainable LoRA and expert/head parameters.")
    return base_params, lora_params


def build_optimizer(model, train_config):
    """Build optimizer groups after configure_stage sets requires_grad flags."""
    base_params, lora_params = split_trainable_parameters(model)
    return AdamW(
        [
            {"params": base_params, "lr": train_config.lr},
            {"params": lora_params, "lr": train_config.lora_lr},
        ],
        weight_decay=train_config.weight_decay,
    )


def build_scheduler(optimizer, train_config):
    """Use linear warmup followed by cosine decay over the current stage."""
    warmup = min(train_config.warmup_epochs, train_config.num_epochs)

    def lr_lambda(epoch):
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, (epoch - warmup + 1) / max(1, train_config.num_epochs - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_model(train_config, tokenizer, adapter_kwargs):
    """Initialize the frozen backbones and trainable MoEA-Net modules."""
    image_token_index = tokenizer.convert_tokens_to_ids("<|image_feature|>")
    model = ViTLlamaModel(
        train_config, False, tokenizer, image_token_index, adapter_kwargs
    ).move_to_device(DEVICE)
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    return model


def stage_epochs(train_config, stage, paper_epochs):
    """Prefer CLI-configured stage epochs, falling back to paper defaults."""
    return stage_epoch_count(train_config.stage_epochs, stage, paper_epochs)


def stage_dataloaders(train_config, modalities, tokenizer, train_batch_size, val_batch_size):
    """Build train/validation loaders for the active modality set."""
    train_dataloader = build_dataloader(
        train_config, modalities, tokenizer, "train", batch_size=train_batch_size
    )
    val_dataloader = None
    if train_config.run_validation:
        val_dataloader = build_dataloader(
            train_config, modalities, tokenizer, "val", batch_size=val_batch_size
        )
    return train_dataloader, val_dataloader


def print_stage_summary(stage, modalities, train_config, train_dataloader, val_dataloader,
                        train_batch_size, val_batch_size):
    """Print the key stage settings before launching a long training run."""
    print(
        f"Stage {stage}: {modalities}, {train_config.num_epochs} epochs, "
        f"train={len(train_dataloader.dataset)}, "
        f"val={len(val_dataloader.dataset) if val_dataloader else 0}, "
        f"train_batch={train_batch_size}, val_batch={val_batch_size}, "
        f"accumulation={train_config.gradient_accumulation_steps}"
    )


def reload_best_stage_checkpoint(model, train_config, modalities, stage):
    """Start the next stage from the best validation checkpoint of this stage."""
    if not getattr(train_config, "save_model", True) or not hasattr(model, "from_pretrained"):
        return
    loaded_path = model.from_pretrained(
        train_config.output_dir,
        "peft_model_best",
        target_modalities=modalities,
        training_stage=stage,
    )
    print(f"Stage {stage}: loaded best checkpoint from {loaded_path}")


def run_stage(model, train_config, tokenizer, peft_config, device, stage, modalities):
    """Train one progressive modality stage and reload its best checkpoint."""
    train_batch_size = stage_batch_size(train_config.stage_batch_sizes, stage, None)
    val_batch_size = stage_batch_size(train_config.stage_batch_sizes, stage, None)
    train_dataloader, val_dataloader = stage_dataloaders(
        train_config, modalities, tokenizer, train_batch_size, val_batch_size
    )

    configure_stage(model, stage, modalities, peft_config)
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    print_stage_summary(
        stage, modalities, train_config, train_dataloader, val_dataloader,
        train_batch_size, val_batch_size
    )

    fit_prognosis(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        train_config,
        device,
    )
    reload_best_stage_checkpoint(model, train_config, modalities, stage)


def main(**kwargs):
    """Fire entry point: CLI arguments override train_config defaults."""
    train_config = TRAIN_CONFIG()
    update_config(train_config, **kwargs)
    validate_training_config(train_config)
    seed_everything(train_config.seed)

    tokenizer = load_prognosis_tokenizer(train_config.tokenizer_name or train_config.model_name)
    model = build_model(train_config, tokenizer, kwargs)
    peft_config = generate_peft_config(train_config, kwargs)

    device = torch.device(DEVICE)
    first_output_dir = train_config.output_dir

    # Stages are nested: OCT, OCT+fundus, then OCT+fundus+FAZ.
    for stage, (modalities, paper_epochs) in enumerate(STAGES, start=1):
        train_config.num_epochs = stage_epochs(train_config, stage, paper_epochs)
        # Keep stage outputs as sibling directories under the same experiment root.
        train_config.output_dir = str(stage_output_dir(first_output_dir, stage))
        run_stage(model, train_config, tokenizer, peft_config, device, stage, modalities)


if __name__ == "__main__":
    fire.Fire(main)
