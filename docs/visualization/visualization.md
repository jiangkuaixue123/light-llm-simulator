# Visualization Tools

Light LLM Simulator provides several visualization tools to analyze search results and deployment configurations.

## Available Tools

### 1. throughput (`throughput.py`)

Visualize throughput changes with number of cards and (kv_len, seq_len)

**Location**: `src/visualization/throughput.py`

**Output**: 
- Generate multiple images showing how throughput varying with total dies under the same kv_len and tpot.
- Generate multiple images showing how throughput varying with kv_len and tpot under the same total dies.

**Usage**:
```bash
python src/visualization/throughput.py
```
