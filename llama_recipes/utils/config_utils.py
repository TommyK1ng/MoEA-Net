"""Configuration helpers used by the prognosis training entry points."""

from dataclasses import asdict

from peft import LoraConfig

from llama_recipes.configs import lora_config, train_config


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for item in config:
            update_config(item, **kwargs)
        return

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif "." in key:
            config_name, attribute = key.split(".", 1)
            if type(config).__name__ == config_name:
                if not hasattr(config, attribute):
                    raise ValueError(f"Unknown {config_name} setting: {attribute}")
                setattr(config, attribute, value)
        elif isinstance(config, train_config):
            print(f"Warning: unknown parameter {key}")


def generate_peft_config(_training_config, kwargs):
    config = lora_config()
    update_config(config, **kwargs)
    return LoraConfig(**asdict(config))
