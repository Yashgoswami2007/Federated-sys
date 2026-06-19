"""
Pydantic schema for fusionnet-client config.yaml validation.

Catches misconfiguration early (typos, missing keys, wrong types) instead of
failing silently deep inside training loops.

Usage:
    from config_schema import load_and_validate_config
    config = load_and_validate_config("config.yaml")
"""

import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    name: str = Field(
        description="HuggingFace model ID. Must match across all federation nodes."
    )
    quantization_type: str = Field(
        default="nf4",
        description="BitsAndBytes quantization type for GPU path."
    )

    @field_validator("quantization_type")
    @classmethod
    def validate_quant_type(cls, v):
        allowed = {"nf4", "fp4", "none"}
        if v not in allowed:
            raise ValueError(f"quantization_type must be one of {allowed}, got '{v}'")
        return v


class HubConfig(BaseModel):
    repo_id: str = Field(description="HuggingFace repo ID for parameter exchange.")
    repo_type: str = Field(default="dataset")


class FederationConfig(BaseModel):
    lora_rank: int = Field(default=8, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    local_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=4, ge=1)
    aggregation_frequency: int = Field(default=1, ge=1)
    target_modules: List[str] = Field(default=["q_proj", "v_proj"])
    hub: Optional[HubConfig] = None

    @field_validator("target_modules")
    @classmethod
    def validate_target_modules(cls, v):
        if not v:
            raise ValueError("target_modules must contain at least one module name.")
        return v


class DatasetConfig(BaseModel):
    name: str = Field(default="banking77")
    text_column: str = Field(default="text")
    label_column: str = Field(default="label")


class PrivacyConfig(BaseModel):
    use_dp_sgd: bool = Field(default=True)
    epsilon: float = Field(default=1.0, gt=0)
    delta: float = Field(default=1e-5, gt=0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0)


class DeviceProfileConfig(BaseModel):
    rank: int = Field(ge=1, le=256)
    batch_size: int = Field(ge=1)


class BackendConfig(BaseModel):
    url: str = Field(default="http://localhost:8000")
    enabled: bool = Field(default=True)
    heartbeat_interval: int = Field(default=30, ge=5)


class FusionNetConfig(BaseModel):
    model: ModelConfig
    federation: FederationConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    device_profiles: Dict[str, DeviceProfileConfig] = Field(default_factory=dict)
    backend: Optional[BackendConfig] = None


def load_and_validate_config(config_path: str) -> dict:
    """Load config.yaml and validate with Pydantic schema.
    
    Returns the validated config as a plain dict (so existing code that uses
    dict access patterns doesn't need to change).
    
    Raises:
        pydantic.ValidationError: If config has invalid values.
        FileNotFoundError: If config_path doesn't exist.
    """
    import yaml

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    validated = FusionNetConfig(**raw_config)
    logger.info(f"Config validated successfully from {config_path}")
    
    # Return as dict so existing code using config["key"] still works
    return validated.model_dump()
