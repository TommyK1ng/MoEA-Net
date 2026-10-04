"""Single-device training and evaluation for the two prognosis heads."""

import csv
import json
import math
from itertools import islice
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def _move_batch(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.no_grad()
def predict_prognosis(model, dataloader, device, name="test"):
    """Run classification heads and return rows ready for metrics/CSV output."""
    model.eval()
    rows = []
    for batch in tqdm(dataloader, desc=f"Evaluating {name}", leave=False):
        batch = _move_batch(batch, device)
        fluid_pred, vision_pred, fluid_probs, vision_probs, _ = model.generate(**batch)
        labels = batch["labels"].cpu().tolist()
        ids = batch["sample_id"].reshape(-1).cpu().tolist()
        fluid_predictions = fluid_pred.cpu().tolist()
        vision_predictions = vision_pred.cpu().tolist()
        fluid_scores = fluid_probs.cpu().tolist()
        vision_scores = vision_probs.reshape(-1).cpu().tolist()
        if len({
            len(ids), len(labels), len(fluid_predictions), len(vision_predictions),
            len(fluid_scores), len(vision_scores),
        }) != 1:
            raise ValueError(f"Prediction count does not match the {name} batch size.")
        for sample_id, target, fluid, vision, fluid_score, vision_score in zip(
            ids, labels, fluid_predictions, vision_predictions, fluid_scores, vision_scores
        ):
            rows.append({
                "sample_id": sample_id,
                "fluid_true": target[0],
                "fluid_pred": fluid,
                "fluid_prob_0": fluid_score[0],
                "fluid_prob_1": fluid_score[1],
                "fluid_prob_2": fluid_score[2],
                "vision_true": target[1],
                "vision_pred": vision,
                "vision_prob": vision_score,
            })
    return rows


def save_evaluation(rows, output_dir, name):
    """Persist predictions and the metrics used for model selection."""
    if not rows:
        raise ValueError("Cannot evaluate an empty dataloader.")

    fluid_true = [row["fluid_true"] for row in rows]
    fluid_pred = [row["fluid_pred"] for row in rows]
    vision_true = [row["vision_true"] for row in rows]
    vision_pred = [row["vision_pred"] for row in rows]
    metrics = {
        "fluid_accuracy": accuracy_score(fluid_true, fluid_pred),
        "fluid_macro_f1": f1_score(
            fluid_true, fluid_pred, labels=[0, 1, 2], average="macro", zero_division=0
        ),
        "vision_accuracy": accuracy_score(vision_true, vision_pred),
        "vision_f1": f1_score(vision_true, vision_pred, zero_division=0),
    }
    metrics["mean_accuracy"] = (metrics["fluid_accuracy"] + metrics["vision_accuracy"]) / 2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"predictions_{name}.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / f"metrics_{name}.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    return metrics


def evaluate_prognosis(model, dataloader, output_dir, name, device):
    rows = predict_prognosis(model, dataloader, device, name)
    return save_evaluation(rows, output_dir, name)


def train_epoch(model, dataloader, optimizer, gradient_accumulation_steps, device,
                use_fp16=False, gradient_clipping_threshold=None, max_steps=None, scaler=None):
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    steps = len(dataloader) if max_steps is None else min(len(dataloader), max_steps)
    if steps < 1:
        raise ValueError("Cannot train with zero batches.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    total_loss = 0.0
    total_samples = 0
    group_samples = 0
    progress = tqdm(islice(dataloader, steps), total=steps, desc="Training", leave=False)
    for step, batch in enumerate(progress):
        batch = _move_batch(batch, device)
        batch_size = next(iter(batch.values())).size(0)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            loss = model(**batch).loss
        batch_loss = loss.detach().float().item()
        if not math.isfinite(batch_loss):
            raise FloatingPointError(f"Non-finite training loss at batch {step + 1}: {batch_loss}")

        total_loss += batch_loss * batch_size
        total_samples += batch_size
        group_samples += batch_size
        # The model returns a batch mean; accumulate sample sums within each group.
        scaler.scale(loss * batch_size).backward()

        if (step + 1) % gradient_accumulation_steps == 0 or step + 1 == steps:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_samples)
            if gradient_clipping_threshold is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0

        progress.set_postfix(
            batch_loss=f"{batch_loss:.4f}",
            avg_loss=f"{total_loss / total_samples:.4f}",
            refresh=False,
        )
    return total_loss / total_samples, steps


def fit_prognosis(model, train_dataloader, val_dataloader, optimizer, scheduler,
                  train_config, device):
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -math.inf
    history = []
    seen_steps = 0
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.use_fp16)
    for epoch in range(1, train_config.num_epochs + 1):
        remaining = (
            None if train_config.max_train_step <= 0
            else train_config.max_train_step - seen_steps
        )
        if remaining is not None and remaining <= 0:
            break
        loss, steps = train_epoch(
            model,
            train_dataloader,
            optimizer,
            train_config.gradient_accumulation_steps,
            device,
            use_fp16=train_config.use_fp16,
            gradient_clipping_threshold=(
                train_config.gradient_clipping_threshold if train_config.gradient_clipping else None
            ),
            max_steps=remaining,
            scaler=scaler,
        )
        seen_steps += steps
        scheduler.step()
        record = {"epoch": epoch, "train_loss": loss}
        if val_dataloader is not None:
            metrics = evaluate_prognosis(
                model, val_dataloader, output_dir, f"val_epoch_{epoch}", device
            )
            record.update(metrics)
            # Validation mean accuracy selects the checkpoint used by the next stage.
            if train_config.save_model and metrics["mean_accuracy"] > best_score:
                model.save_pretrained(output_dir, "peft_model_best")
            best_score = max(best_score, metrics["mean_accuracy"])
        elif train_config.save_model:
            model.save_pretrained(output_dir, "peft_model_best")
        history.append(record)
        print(f"Epoch {epoch}: {record}")
    if train_config.save_metrics:
        with (output_dir / "history.json").open("w", encoding="utf-8") as history_file:
            json.dump(history, history_file, indent=2)
    return history
