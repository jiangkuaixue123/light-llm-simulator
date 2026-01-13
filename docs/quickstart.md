# QuickStart

This guide will help you quickly get started with Light LLM Simulator.

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/light-llm-simulator.git
cd light-llm-simulator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Simulator

- Run with default settings (DeepSeek-V3, latency 50-200ms, micro_batch_num 2/3):

    ```bash
    python src/cli/main.py
    ```

 - Run with Customize Parameters

    ```bash
    python src/cli/main.py
    ```

### 4. Visualization

```bash
# Visualize throughput changes
python src/visualization/throughput.py
```

## Parameter Explanation

### Simulator Parameter Explanation

#### `serving_mode`
- The serving mode of the task
- type: str
- Default: `AFD`
- Supported Serving mode: `AFD`, `DeepEP`

#### `--model_type`
- Model identifier (see `ModelType` enum)
- type: str
- Default: `"deepseek-ai/DeepSeek-V3"`
- Supported Model Type: `"deepseek-ai/DeepSeek-V3"`, `"Qwen/Qwen3-235B-A22B"`

#### `--device_type`
- Device identifier (see `DeviceType` enum)
- type: str
- Default: `"Ascend_A3Pod"`
- Supported Device Type:
    - **Ascend**: 910B2, 910B3, 910B4, A3Pod, David121, David120
    - **Nvidia**: A100SXM, H100SXM

#### `--min_attn_bs`
- The min number of attention batch size to explore.
- type: int
- Default: 2

#### `--max_attn_bs`
- The max number of attention batch size to explore.
- type: int
- Default: 1000

#### `--min_die`
- The min number of die to explore.
- type: int
- Default: 16

#### `--max_die`
- The max number of die to explore.
- type: int
- Default: 768

#### `--topt`
- The target TPOT.
- type: list[int]
- Default: [20, 50, 70, 100, 150]

#### `--kv_len`
- The input sequence length.
- type: list[int]
- Default: [2048, 4096, 8192, 16384, 131072]

#### `--micro_batch_num`
- The micro batch number.
- type: list[int]
- Default: [2, 3]

#### `--next_n`
- Predict the next n tokens through the MTP(Multi-Token Prediction) technique.
- type: int
- Default: 1

#### `--multi_token_ratio`
- The acceptance rate of the additionally predicted token.
- type: float
- Default: 0.7

#### `--attn_tensor_parallel`
- Number of dies used for tensor model parallelism.
- type: int
- Default: 1

#### `--ffn_tensor_parallel`
- Number of dies used for tensor model parallelism.
- type: int
- Default: 1

### Visualization Parameter Explanation

#### `--serving_mode`
- The serving mode of the task
- type: str
- Default: `AFD`
- Supported Serving mode: `AFD`, `DeepEP`

#### `--model_type`
- Model identifier (see `ModelType` enum)
- type: str
- Default: `"deepseek-ai/DeepSeek-V3"`
- Supported Model Type: `"deepseek-ai/DeepSeek-V3"`, `"Qwen/Qwen3-235B-A22B"`

#### `--device_type`
- Device identifier (see `DeviceType` enum)
- type: str
- Default: `"Ascend_A3Pod"`
- Supported Device Type:
    - **Ascend**: 910B2, 910B3, 910B4, A3Pod, David121, David120
    - **Nvidia**: A100SXM, H100SXM

#### `--topt_list`
- The target TPOT.
- type: list[int]
- Default: [20, 50, 70, 100, 150]

#### `--kv_len_list`
- The input sequence length.
- type: list[int]
- Default: [2048, 4096, 8192, 16384, 131072]

#### `--total_die`
- The number of total die.
- type: list[int]
- Default: [64]

#### `--micro_batch_num`
- The micro batch number.
- type: list[int]
- Default: [2, 3]

#### `--min_die`
- The min number of die to explore.
- type: int
- Default: 0

#### `--max_die`
- The max number of die to explore.
- type: int
- Default: 768
