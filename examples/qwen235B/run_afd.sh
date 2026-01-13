# run in INFO level to get output.log
rm -rf ./output.log
cur_date=$(LC_ALL=en_US.utf8 date + "%b%d-%H%M") && echo $cur_date

export LOG_LEVEL=INFO
python examples/qwen235B/afd.py > data/output-${cur_date}.log 2>&1
ln -s data/output-${cur_date}.log ./output.log
echo "saved perf datas to data/output-${cur_date}.log"

# Visualize throughput changes
python src/visualization/throughput.py \
--serving_mode "AFD" \
--model_type "Qwen/Qwen3-235B-A22B" \
--device_type "Ascend_A3Pod" \
--tpot_list 50 \
--kv_len_list 4096 \
--micro_batch_num 3