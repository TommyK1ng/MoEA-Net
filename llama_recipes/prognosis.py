"""Shared stage and tokenizer settings for retinal prognosis experiments."""

from pathlib import Path


# Stage order also defines modality-to-expert assignment.
STAGES = (
    (("oct",), 60),
    (("oct", "fundus"), 50),
    (("oct", "fundus", "faz"), 40),
)
SPECIAL_TOKENS = (
    "<|image_feature|>",
    "[Img]",
    "[/Img]",
    "[MRG]",
    "<|vision_change|>",
    "<|fluid_change|>",
)


def stage_output_dir(first_output_dir, stage):
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    first = Path(first_output_dir)
    return first if stage == 1 else first.parent / ("Second_Stage" if stage == 2 else "Third_Stage")


def _stage_tuple_value(values, stage, fallback, name):
    if values is None:
        return fallback
    if len(values) != len(STAGES) or any(type(value) is not int or value < 1 for value in values):
        raise ValueError(f"{name} must contain three positive integers.")
    return values[stage - 1]


def stage_batch_size(batch_sizes, stage, fallback):
    return _stage_tuple_value(batch_sizes, stage, fallback, "stage_batch_sizes")


def stage_epoch_count(epoch_counts, stage, fallback):
    return _stage_tuple_value(epoch_counts, stage, fallback, "stage_epochs")


def load_prognosis_tokenizer(model_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # These tokens are trained as special embeddings and used as visual/task slots.
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    return tokenizer
