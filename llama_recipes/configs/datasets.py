# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

@dataclass
class oct_eye_dataset:
    dataset: str = "oct_eye_dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
