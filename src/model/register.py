from conf.model_config import ModelType
from conf.config import Config
from src.model.base import BaseModule
from src.model.deepseekv3_decode import (
    DeepSeekV3DecodeAttn,
    DeepSeekV3DecodeMLP,
    DeepSeekV3DecodeMoe,
)
from src.model.qwen235_decode import (
    Qwen235DecodeAttn,
    Qwen235DecodeMoe,
)


def get_model(
    config: Config
)-> BaseModule:
    '''
    Description:
        Get all modules of the specified model.
    Args:
        config: The configuration of the search task.
    Returns:
        A dictionary that contains all modules of the specified model.
    '''
    assert(config.model_type in ModelType), f"unsupport model {config.model_type}"

    if config.model_type == ModelType.DEEPSEEK_V3:
        attn = DeepSeekV3DecodeAttn(config)
        mlp = DeepSeekV3DecodeMLP(config)
        moe = DeepSeekV3DecodeMoe(config)
        model = {"attn": attn, "mlp": mlp, "moe": moe}
    elif config.model_type.name.startswith("QWEN3_235B"):
        attn = Qwen235DecodeAttn(config)
        moe = Qwen235DecodeMoe(config)
        model = {"attn": attn, "moe": moe}
    return model

def get_attention_family(
    model_type: str,
)-> str:
    '''
    Description:
        Get the attention mechanism of the specified model.
    Args:
        model_type: The type of the model.
    Returns:
        The attention mechanism of the specified model.
    '''
    assert(model_type in ModelType), f"unsupport model {model_type}"
    if model_type == ModelType.DEEPSEEK_V3:
        return "MLA"
    if model_type.name.startswith("QWEN3_235B"):
        return "GQA"
