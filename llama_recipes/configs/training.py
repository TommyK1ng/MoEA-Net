"""Configuration for the three-stage MoEA-Net prognosis pipeline."""

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer_name: str | None = None
    dataset_directory: str = "data/Splits"
    image_directory: str | None = None
    vision_encoder_path: str | None = None
    output_dir: str = "runs/moeanet/First_Stage"

    stage_batch_sizes: tuple[int, int, int] = (32, 32, 32)
    stage_epochs: tuple[int, int, int] = (60, 50, 40)
    num_workers_dataloader: int = 1
    oversample_train: bool = True

    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    lora_lr: float = 2e-5
    fluid_loss_weight: float = 1.0
    vision_loss_weight: float = 2.0
    expert_temperature: float = 0.6
    weight_decay: float = 0.0
    warmup_epochs: int = 4
    seed: int = 45

    quantization: bool = True
    use_fp16: bool = False
    use_fast_kernels: bool = False
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    max_train_step: int = 0

    run_validation: bool = True
    save_model: bool = True
    save_metrics: bool = False
