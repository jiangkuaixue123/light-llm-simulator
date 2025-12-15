# AFD (Attention-FFN Disaggregated) Search

AFD is a deployment strategy that disaggregates Attention and FFN computations across different hardware resources to optimize throughput while meeting latency targets.

## Overview

In AFD serving, Attention and FFN workers are separated, allowing independent scaling and optimization. The search algorithm finds optimal configurations by:

1. Determining the maximum attention batch size that satisfies latency and memory targets
2. Exploring different combinations of Attention and FFN die allocations
3. Selecting configurations with highest throughput, deduplicating by total die and sorting by throughput

## Output Columns

The search outputs the following columns:

- `attn_bs`: Attention batch size
- `ffn_bs`: FFN batch size
- `kv_len`: KV cache length
- `attn_die`: Number of attention dies
- `ffn_die`: Number of FFN dies
- `total_die`: Number of total dies
- `attn_time`: Attention time for per layer (μs)
- `ffn_time`: FFN time for per layer (μs)
- `commu_time`: communication time per layer(μs)
- `e2e_time`: End-to-end time (ms)
- `e2e_time_per_dense_layer`: End-to-end time for per dense layers (μs)
- `e2e_time_per_moe_layer`: End-to-end time for per MoE layers (μs)
- `throughput`: Throughput (tokens/second)

The above output results are cached in CSV files for analysis

## Usage

```python
from src.search.afd import AfdSearch

config = Config(
   serving_mode,
   model_type,
   device_type,
   min_attn_bs,
   max_attn_bs,
   min_die,
   max_die,
   tpot,
   kv_len,
   micro_batch_num,
   next_n,
   multi_token_ratio,
   attn_tensor_parallel,
   ffn_tensor_parallel
)
afd_search = AfdSearch(config)
afd_search.deployment()
```

## Targets

### Latency Targets

1. **Attention Module Latency**:
   ```
   micro_batch_num * attn_time < latency / num_layers * (1 + multi_token_ratio)
   ```

2. **MoE Module Latency**:
    ```
    micro_batch_num * moe_time < latency / num_layers * (1 + multi_token_ratio)
    ```
3. **MoE Layer Latency**:
   ```
   e2e_time_per_moe_layer = max(attn_time + moe_time + commu_time,max(attn_time, moe_time) * self.micro_batch_num)

   e2e_time_per_moe_layer * num_layers < latency * (1 + multi_token_ratio)
   ```

### Memory Targets

1. **Attention Memory**:
   ```
   kv_size * micro_batch_num + attn_static_memory < npu_memory * MEMORY_THRESHOLD_RATIO
   ```

2. **FFN Static Memory**:
   ```
   ffn_static_memory + ffn dynamic memory < npu_memory * MEMORY_THRESHOLD_RATIO
   ```
