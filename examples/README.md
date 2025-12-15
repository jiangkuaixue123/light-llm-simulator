## Examples

This directory contains runnable examples for **Light LLM Simulator**.

### Layout

```
├── examples/    # runnable examples
│   ├── deepseek/    # deepseek
│   │   ├── afd.py    # Python example that runs AFD
│   │   ├── deepep.py    # Python example that runs DeepEP
│   │   ├── run_afd.sh    # Convenience shell script to run the AFD example
│   │   ├── run_deepep.sh    # Convenience shell script to run the DeepEP example
│   ├── qwen235B/      # qwen235B
│   │   ├── afd.py    # Python example that runs AFD
│   │   ├── deepep.py    # Python example that runs DeepEP
│   │   ├── run_afd.sh    # Convenience shell script to run the AFD example
│   │   ├── run_deepep.sh    # Convenience shell script to run the DeepEP example
```
### DeepSeek Example

#### Run with the helper shell script

##### AFD
```bash
  bash examples/deepseek/run_afd.sh
```
This script will:
- Set `LOG_LEVEL=INFO`.
- Run the `deepseek/afd.py` example
    - Construct a **DeepSeek-V3** model config.
    - Use **Ascend A3Pod ** as the hardware topology.
    - Run **AFD** search once with:
        - `tpot = 50 ms`
        - `micro_batch_num = 2`
        - `kv_len = 4096`
        - `seq_len = 1 + next_n` (with `next_n = 1`)
- Save logs to:
  - `data/output-<timestamp>.log`
  - Also create a symlink `output.log` pointing to the latest run.
- Run visualization scripts:
  - `src/visualization/pareto.py` (Pareto frontier plots)
  - `src/visualization/pipeline.py` (AFD pipeline visualization)

##### DeepEP
```bash
  bash examples/deepseek/run_deepep.sh
```
This script will:
- Set `LOG_LEVEL=INFO`.
- Run the `deepseek/deepep.py` example
    - Construct a **DeepSeek-V3** model config.
    - Use **Ascend A3Pod ** as the hardware topology.
    - Run **DeepEP** search once with:
        - `tpot = 50 ms`
        - `micro_batch_num = 1`
        - `kv_len = 4096`
        - `seq_len = 1 + next_n` (with `next_n = 1`)
- Save logs to:
  - `data/output-<timestamp>.log`
  - Also create a symlink `output.log` pointing to the latest run.
- Run visualization scripts:
  - `src/visualization/pareto.py` (Pareto frontier plots)

### DeepSeek Example

#### Run with the helper shell script

##### AFD
```bash
  bash examples/qwen235B/run_afd.sh
```
This script will:
- Set `LOG_LEVEL=INFO`.
- Run the `qwen235B/afd.py` example
    - Construct a **QWEN3-235B-A22B** model config.
    - Use **Ascend A3Pod ** as the hardware topology.
    - Run **AFD** search once with:
        - `tpot = 50 ms`
        - `micro_batch_num = 2`
        - `kv_len = 4096`
        - `seq_len = 1 + next_n` (with `next_n = 1`)
- Save logs to:
  - `data/output-<timestamp>.log`
  - Also create a symlink `output.log` pointing to the latest run.
- Run visualization scripts:
  - `src/visualization/pareto.py` (Pareto frontier plots)
  - `src/visualization/pipeline.py` (AFD pipeline visualization)

##### DeepEP
```bash
  bash examples/qwen235B/run_deepep.sh
```
This script will:
- Set `LOG_LEVEL=INFO`.
- Run the `qwen235B/deepep.py` example
    - Construct a **QWEN3-235B-A22B** model config.
    - Use **Ascend A3Pod ** as the hardware topology.
    - Run **DeepEP** search once with:
        - `tpot = 50 ms`
        - `micro_batch_num = 1`
        - `kv_len = 4096`
        - `seq_len = 1 + next_n` (with `next_n = 1`)
- Save logs to:
  - `data/output-<timestamp>.log`
  - Also create a symlink `output.log` pointing to the latest run.
- Run visualization scripts:
  - `src/visualization/pareto.py` (Pareto frontier plots)
