"""Paired retinal images and prognosis labels for MoEA-Net."""

import json
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .augmentation import PairedRetinalAugmenter


IMAGE_TOKEN = "<|image_feature|>"
FLUID_LABELS = {"不吸收": 0, "部分吸收": 1, "吸收": 2}
VISION_LABELS = {"不提升": 0, "提升": 1}
FLUID_PATTERN = re.compile(r"视网膜下积液变化：([^;；]+)[;；]")
VISION_PATTERN = re.compile(r"视力提升：([^。]+)。(?:$|\s)")
PROMPTS = (
    "Given one modality data that includes pre-treatment, post-treatment, and differential feature, please assist in predicting subretinal fluid change and visual acuity improvement.",
    "Based on two modalities of medical data, each comprising pre-treatment, post-treatment, and difference feature, please assist in predicting subretinal fluid change and visual acuity improvement.",
    "Given three different modalities of data, each comprising pre-treatment, post-treatment, and the difference feature, please assist in predicting subretinal fluid change and visual acuity improvement."
)


class OctEyeDataset(Dataset):
    """Return [modalities, before/after, 3, 224, 224] images and two labels."""

    def __init__(self, dataset_directory, target_modalities, tokenizer, train_config, split="train"):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if not 1 <= len(target_modalities) <= 3 or len(set(target_modalities)) != len(target_modalities):
            raise ValueError("Expected one to three distinct modalities in model order.")

        split_dir = Path(dataset_directory)
        image_directory = getattr(train_config, "image_directory", None)
        self.image_directory = Path(image_directory) if image_directory else split_dir.parent / "Samples"
        with (split_dir / f"{split}.json").open(encoding="utf-8") as split_file:
            samples = json.load(split_file)

        # A sample belongs to a stage only if it contains every required modality.
        required = set(target_modalities)
        self.samples = [
            sample for sample in samples
            if required.issubset(sample.get("modalitys", []))
        ]
        if split == "train" and getattr(train_config, "oversample_train", True):
            self.samples = self._oversample_minority_classes(self.samples)
        self.target_modalities = tuple(target_modalities)
        self.is_train = split == "train"
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.augmenter = PairedRetinalAugmenter() if self.is_train else None
        count = len(self.target_modalities)
        # Each modality contributes three visual tokens: before, after, and difference.
        prompt = (
            "[Img]" + IMAGE_TOKEN * (3 * count) + "[/Img][MRG]"
            + PROMPTS[count - 1] + "<|fluid_change|><|vision_change|>"
        )
        self.input_ids = torch.tensor(
            tokenizer.encode(tokenizer.bos_token + prompt, add_special_tokens=False),
            dtype=torch.long,
        )

    @staticmethod
    def _extract_labels(findings):
        fluid_match = FLUID_PATTERN.search(findings)
        vision_match = VISION_PATTERN.search(findings)
        fluid = FLUID_LABELS.get(fluid_match.group(1).strip()) if fluid_match else None
        vision = VISION_LABELS.get(vision_match.group(1).strip()) if vision_match else None
        if fluid is None or vision is None:
            raise ValueError(f"Invalid prognosis findings: {findings}")
        return fluid, vision

    @classmethod
    def _oversample_minority_classes(cls, samples):
        if not samples:
            return samples

        buckets = {}
        for sample in samples:
            label_key = cls._extract_labels(sample["findings"])
            buckets.setdefault(label_key, []).append(sample)

        target_count = max(len(bucket) for bucket in buckets.values())
        balanced = []
        for label_key in sorted(buckets):
            bucket = buckets[label_key]
            repeats, remainder = divmod(target_count, len(bucket))
            balanced.extend(bucket * repeats)
            balanced.extend(bucket[:remainder])
        return balanced

    def _load_pair(self, sample_id, modality):
        if len(sample_id) < 3:
            raise ValueError(f"Invalid sample_id: {sample_id}")
        folder_id = f"{sample_id[:-2]}_{sample_id[-2]}_{sample_id[-1]}"
        directory = self.image_directory / folder_id
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")

        pattern = re.compile(
            rf"^{re.escape(sample_id)}_(?:os|od)_{re.escape(modality)}_(before|after)\.jpg$",
            re.IGNORECASE,
        )
        paths = {}
        for path in directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                phase = match.group(1).lower()
                if phase in paths:
                    raise ValueError(f"Multiple {phase} images for {sample_id}/{modality}")
                paths[phase] = path
        if set(paths) != {"before", "after"}:
            raise FileNotFoundError(f"Incomplete image pair for {sample_id}/{modality} in {directory}")

        images = []
        for phase in ("before", "after"):
            with Image.open(paths[phase]) as image:
                images.append(self.image_transform(image.convert("RGB")))
        return torch.stack(images)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_id = str(sample["sample_id"])
        pairs = []
        for modality in self.target_modalities:
            images = self._load_pair(sample_id, modality)
            if self.augmenter is not None:
                # Keep pre- and post-treatment images spatially aligned after augmentation.
                images = self.augmenter(images)
            pairs.append(torch.stack([self.normalize(image) for image in images]))
        return {
            "sample_id": torch.tensor([int(sample_id)], dtype=torch.long),
            "input_ids": self.input_ids.clone(),
            "attention_mask": torch.ones_like(self.input_ids),
            "modality_image_pairs": torch.stack(pairs),
            "labels": torch.tensor(self._extract_labels(sample["findings"]), dtype=torch.long),
        }
