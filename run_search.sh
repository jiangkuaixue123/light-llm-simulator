#!/bin/bash

# for E in 2 4 6 8; do
# 并行执行所有搜索任务，每个任务作为一个后台进程并限制并发数量，充分利用多核多线程
# 设置最大并行任务数，根据机器CPU核心和内存调整。比如并行 8 个任务：
MAX_JOBS=8

# 启动计数
job_count=0

for E in 2 4 6 8 12 16 32; do
  for L in 30 50 70 94; do
    python src/cli/main.py --serving_mode AFD --model_type Qwen/Qwen3-235B-A22B-E${E}-L${L} &
    job_count=$((job_count+1))
    if [[ $job_count -ge $MAX_JOBS ]]; then
      wait -n      # 等待任一后台任务结束，释放一个槽位
      job_count=$((job_count-1))
    fi

    python src/cli/main.py --serving_mode DeepEP --model_type Qwen/Qwen3-235B-A22B-E${E}-L${L} &
    job_count=$((job_count+1))
    if [[ $job_count -ge $MAX_JOBS ]]; then
      wait -n
      job_count=$((job_count-1))
    fi
  done
done

# 等待所有剩余任务
wait