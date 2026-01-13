"""
Model Configuration Module

This module provides configuration management for various large language models (LLMs).
It defines model types, their detailed configurations
including architectural parameters, file paths, and model-specific settings.

The module supports multiple model families:
- DeepSeek V3
- Qwen3-235B-A22B

Key features:
- Enum-based model type definitions for type safety
- Automatic tokenizer selection based on model type
- Centralized model configuration with architectural parameters
- File path generation for model checkpoints and configurations
- Factory pattern for creating model configurations

Usage:
    from model_config import ModelType, ModelConfig

    model_type = ModelType.DEEPSEEK_V3
    config = ModelConfig.create(model_type)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ModelType(Enum):
    """
    Enumeration of supported model types with their HuggingFace model identifiers.

    This enum provides a centralized way to reference different LLM models
    and ensures type safety when working with model configurations.
    """
    DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
    QWEN3_235B = "Qwen/Qwen3-235B-A22B"
    QWEN3_235B_E2_L30 = "Qwen/Qwen3-235B-A22B-E2-L30"
    QWEN3_235B_E2_L50 = "Qwen/Qwen3-235B-A22B-E2-L50"
    QWEN3_235B_E2_L70 = "Qwen/Qwen3-235B-A22B-E2-L70"
    QWEN3_235B_E4_L30 = "Qwen/Qwen3-235B-A22B-E4-L30"
    QWEN3_235B_E4_L50 = "Qwen/Qwen3-235B-A22B-E4-L50"
    QWEN3_235B_E4_L70 = "Qwen/Qwen3-235B-A22B-E4-L70"
    QWEN3_235B_E6_L30 = "Qwen/Qwen3-235B-A22B-E6-L30"
    QWEN3_235B_E6_L50 = "Qwen/Qwen3-235B-A22B-E6-L50"
    QWEN3_235B_E6_L70 = "Qwen/Qwen3-235B-A22B-E6-L70"
    QWEN3_235B_E8_L30 = "Qwen/Qwen3-235B-A22B-E8-L30"
    QWEN3_235B_E8_L50 = "Qwen/Qwen3-235B-A22B-E8-L50"
    QWEN3_235B_E8_L70 = "Qwen/Qwen3-235B-A22B-E8-L70"
    QWEN3_235B_E2_L94 = "Qwen/Qwen3-235B-A22B-E2-L94"
    QWEN3_235B_E4_L94 = "Qwen/Qwen3-235B-A22B-E4-L94"
    QWEN3_235B_E6_L94 = "Qwen/Qwen3-235B-A22B-E6-L94"
    QWEN3_235B_E8_L94 = "Qwen/Qwen3-235B-A22B-E8-L94"
    QWEN3_235B_E12_L30 = "Qwen/Qwen3-235B-A22B-E12-L30"
    QWEN3_235B_E12_L50 = "Qwen/Qwen3-235B-A22B-E12-L50"
    QWEN3_235B_E12_L70 = "Qwen/Qwen3-235B-A22B-E12-L70"
    QWEN3_235B_E12_L94 = "Qwen/Qwen3-235B-A22B-E12-L94"
    QWEN3_235B_E16_L30 = "Qwen/Qwen3-235B-A22B-E16-L30"
    QWEN3_235B_E16_L50 = "Qwen/Qwen3-235B-A22B-E16-L50"
    QWEN3_235B_E16_L70 = "Qwen/Qwen3-235B-A22B-E16-L70"
    QWEN3_235B_E16_L94 = "Qwen/Qwen3-235B-A22B-E16-L94"
    QWEN3_235B_E32_L30 = "Qwen/Qwen3-235B-A22B-E32-L30"
    QWEN3_235B_E32_L50 = "Qwen/Qwen3-235B-A22B-E32-L50"
    QWEN3_235B_E32_L70 = "Qwen/Qwen3-235B-A22B-E32-L70"
    QWEN3_235B_E32_L94 = "Qwen/Qwen3-235B-A22B-E32-L94"


@dataclass
class ModelConfig:
    """
    Configuration dataclass containing all architectural parameters for supported models.

    This class stores comprehensive model specifications including:
    - Basic architecture parameters (hidden_size, num_layers, etc.)
    - Attention mechanism configuration (num_heads, kv_heads, head_size)
    - Vocabulary and sequence length settings
    - Model-specific parameters (especially for MoE models like DeepSeek V3)
    - Memory and quantization settings

    Attributes:
        model_type (ModelType): The type of the model
        hidden_size (int): Hidden dimension size of the model
        num_layers (int): Number of transformer layers
        max_kv_length (int): Maximum sequence length the model can handle
        num_heads (int): Number of attention heads
        kv_heads (int): Number of key-value heads (for GQA/MQA)
        vocab_size (int): Size of the vocabulary
        head_size (int): Size of each attention head
        model_size_b (float): Model size in billions of parameters
        intermediate_size (int): Size of the feed-forward network intermediate layer
        chat_template (Optional[Any]): Template for chat-based interactions

        # DeepSeek V3 specific parameters:
        kv_lora_rank (int): Rank for key-value LoRA decomposition
        max_position_embeddings (int): Maximum position embeddings
        moe_intermediate_size (int): MoE intermediate layer size
        n_routed_experts (int): Number of routed experts in MoE
        n_shared_experts (int): Number of shared experts in MoE
        num_attention_heads (int): Number of attention heads (DeepSeek naming)
        num_experts_per_tok (int): Number of experts activated per token
        num_key_value_heads (int): Number of key-value heads (DeepSeek naming)
        first_k_dense_replace (int): Number of initial layers that are dense (not MoE)
        q_lora_rank (int): Rank for query LoRA decomposition
        qk_nope_head_dim (int): Query-key head dimension without positional encoding
        qk_rope_head_dim (int): Query-key head dimension with RoPE
        v_head_dim (int): Value head dimension

        # Memory/quantization parameters:
        size_of_w (int): Size multiplier for weights
        size_of_a (int): Size multiplier for activations
        size_of_kv (int): Size multiplier for key-value cache
    """
    model_type: ModelType
    hidden_size: int = 0
    num_layers: int = 0
    max_kv_length: int = 0
    num_heads: int = 0
    kv_heads: int = 0
    vocab_size: int = 0
    head_size: int = 0
    model_size_b: float = 0.0
    intermediate_size: int = 0
    chat_template: Optional[Any] = None
    kv_lora_rank: int = 0
    max_position_embeddings: int = 0
    moe_intermediate_size: int = 0
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    num_attention_heads: int = 0
    num_experts_per_tok: int = 0
    num_key_value_heads: int = 0
    num_moe_layers: int = 0
    first_k_dense_replace: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0

    @classmethod
    def create_model_config(cls, model_type: ModelType) -> 'ModelConfig':
        """
        Factory method: return a ModelConfig instance pre-filled with
        vendor-published model specifications for the requested model type.

        Parameters
        ----------
        model_type : ModelType
            Enum member identifying the target model.
        """
        def cfg(**kwargs):
            return kwargs
        configs = {
            ModelType.DEEPSEEK_V3: cfg(
                model_size_b=671, hidden_size=7168, max_kv_length=128000, intermediate_size=18432,
                kv_lora_rank=512, max_position_embeddings=163840, moe_intermediate_size=2048,
                first_k_dense_replace=3, n_routed_experts=256, n_shared_experts=1,
                num_attention_heads=128, num_experts_per_tok=8, num_layers=61, num_moe_layers=58,
                num_key_value_heads=128, q_lora_rank=1536, qk_nope_head_dim=128,
                qk_rope_head_dim=64, v_head_dim=128, vocab_size=129280),
            ModelType.QWEN3_235B: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=8, num_layers=94, num_moe_layers=94,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E2_L30: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=2, num_layers=30, num_moe_layers=30,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E2_L50: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=2, num_layers=50, num_moe_layers=50,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E2_L70: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=2, num_layers=70, num_moe_layers=70,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E2_L94: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=2, num_layers=94, num_moe_layers=94,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E4_L30: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=4, num_layers=30, num_moe_layers=30,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E4_L50: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=4, num_layers=50, num_moe_layers=50,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E4_L70: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=4, num_layers=70, num_moe_layers=70,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E4_L94: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=4, num_layers=94, num_moe_layers=94,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E6_L30: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=6, num_layers=30, num_moe_layers=30,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E6_L50: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=6, num_layers=50, num_moe_layers=50,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E6_L70: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=6, num_layers=70, num_moe_layers=70,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E6_L94: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=6, num_layers=94, num_moe_layers=94,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E8_L30: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=8, num_layers=30, num_moe_layers=30,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E8_L50: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=8, num_layers=50, num_moe_layers=50,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E8_L70: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=8, num_layers=70, num_moe_layers=70,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E8_L94: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=8, num_layers=94, num_moe_layers=94,
                head_size=128, vocab_size=151936),
            # 新增 E12
            ModelType.QWEN3_235B_E12_L30: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=12, num_layers=30, num_moe_layers=30,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E12_L50: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=12, num_layers=50, num_moe_layers=50,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E12_L70: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=12, num_layers=70, num_moe_layers=70,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E12_L94: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=12, num_layers=94, num_moe_layers=94,
                head_size=128, vocab_size=151936),
            # 新增 E16
            ModelType.QWEN3_235B_E16_L30: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=16, num_layers=30, num_moe_layers=30,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E16_L50: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=16, num_layers=50, num_moe_layers=50,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E16_L70: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=16, num_layers=70, num_moe_layers=70,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E16_L94: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=16, num_layers=94, num_moe_layers=94,
                head_size=128, vocab_size=151936),
            # 新增 E32
            ModelType.QWEN3_235B_E32_L30: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=32, num_layers=30, num_moe_layers=30,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E32_L50: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=32, num_layers=50, num_moe_layers=50,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E32_L70: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=32, num_layers=70, num_moe_layers=70,
                head_size=128, vocab_size=151936),
            ModelType.QWEN3_235B_E32_L94: cfg(
                model_size_b=235, hidden_size=4096, max_kv_length=32768, intermediate_size=12288,
                max_position_embeddings=40960, moe_intermediate_size=1536, n_routed_experts=128,
                num_heads=64, kv_heads=4, num_experts_per_tok=32, num_layers=94, num_moe_layers=94,
                head_size=128, vocab_size=151936),
        }
        try:
            return cls(model_type=model_type, **configs[model_type])
        except KeyError as e:
            raise ValueError(f'Unsupported model_type: {model_type}') from e
