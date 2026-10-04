"""Evaluate nested modality stages and keep the highest-stage prediction per sample."""

import sys
from pathlib import Path

import fire
import torch

# Allow direct execution from recipes/moeanet while importing project modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.datasets import OctEyeDataset
from llama_recipes.models.moeanet.modeling_moeanet import ViTLlamaModel
from llama_recipes.prognosis import STAGES, load_prognosis_tokenizer, stage_batch_size, stage_output_dir
from llama_recipes.utils.config_utils import update_config
from llama_recipes.utils.prognosis_training import (
    predict_prognosis,
    save_evaluation,
)


def merge_stage_predictions(predictions, rows, stage, previous_ids):
    """Replace lower-stage predictions with the highest available modality stage."""
    stage_predictions = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in stage_predictions:
            raise ValueError(f"Duplicate test sample_id: {sample_id}")
        if previous_ids is not None and sample_id not in previous_ids:
            raise ValueError(f"Stage {stage} sample {sample_id} is absent from stage {stage - 1}.")
        earlier = predictions.get(sample_id)
        if earlier is not None and (
            row["fluid_true"], row["vision_true"]
        ) != (earlier["fluid_true"], earlier["vision_true"]):
            raise ValueError(f"Inconsistent test labels for sample {sample_id}.")
        stage_predictions[sample_id] = {**row, "stage": stage}

    predictions.update(stage_predictions)
    return set(stage_predictions)


def main(**kwargs):
    """Evaluate every stage and merge predictions by highest available modality count."""
    train_config = TRAIN_CONFIG()
    # The same CLI fields used for training define model paths and batch sizes here.
    update_config(train_config, **kwargs)
    if not torch.cuda.is_available():
        raise RuntimeError("This 8B model requires a CUDA GPU for evaluation.")

    tokenizer = load_prognosis_tokenizer(
        train_config.tokenizer_name or train_config.model_name
    )
    # Build all nested test subsets so later stages can overwrite earlier results.
    datasets = [
        OctEyeDataset(
            train_config.dataset_directory, list(modalities), tokenizer, train_config, "test"
        )
        for modalities, _ in STAGES
    ]
    if not datasets[0]:
        raise ValueError("No OCT samples found in the test split.")

    # Initialize the architecture once; each loop iteration loads stage-specific weights.
    model = ViTLlamaModel(
        train_config,
        use_cache=False,
        tokenizer=tokenizer,
        image_token_index=tokenizer.convert_tokens_to_ids("<|image_feature|>"),
        adapter_kwargs=kwargs,
    ).move_to_device("cuda:0")
    first_output_dir = train_config.output_dir
    # Test artifacts are written next to the stage checkpoint directories.
    results_dir = Path(first_output_dir).parent / "test_results"
    device = torch.device("cuda:0")
    predictions = {}
    # Tracks the nested sample set so Stage 2/3 cannot introduce unrelated samples.
    previous_ids = None

    for stage, ((modalities, _), dataset) in enumerate(zip(STAGES, datasets), start=1):
        if not dataset:
            previous_ids = set()
            print(f"Stage {stage}: no test samples; skipped.")
            continue

        # Each stage loads its own best checkpoint from First/Second/Third_Stage.
        model.from_pretrained(
            stage_output_dir(first_output_dir, stage),
            peft_model_name="peft_model_best",
            target_modalities=modalities,
            training_stage=stage,
        )
        batch_size = stage_batch_size(train_config.stage_batch_sizes, stage, None)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
        )
        rows = predict_prognosis(model, dataloader, device, name=f"stage_{stage}")
        if len(rows) != len(dataset):
            raise ValueError(
                f"Stage {stage} produced {len(rows)} predictions for "
                f"{len(dataset)} test samples."
            )
        # Later stages overwrite earlier predictions for samples with more modalities.
        previous_ids = merge_stage_predictions(predictions, rows, stage, previous_ids)
        metrics = save_evaluation(rows, results_dir, f"stage_{stage}")
        print(f"Stage {stage}: {len(rows)} samples, mean accuracy {metrics['mean_accuracy']:.3f}")

    # The final report contains one prediction per sample, from its richest stage.
    final_metrics = save_evaluation(list(predictions.values()), results_dir, "test")
    print(
        f"Final test: {len(predictions)} unique samples, "
        f"mean accuracy {final_metrics['mean_accuracy']:.3f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
