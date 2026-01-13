import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import re

color_map = {
    'attn': '#1f77b4',
    'dispatch': '#ff7f0e',
    'moe': '#2ca02c',
    'combine': '#d62728'
}

TARGET_DIES = [32, 64, 128, 256, 288, 384]

def create_gantt_chart(mbn, attn_time, dispatch_time, moe_time, combine_time, total_die, model_name, tpot, kv_len, attn_bs, throughput, e2e_time=0, attn_die=None, ffn_die=None):
    start_time, attn_tmp, dispatch_tmp, ffn_tmp = 0.0, 0.0, 0.0, 0.0
    fig, ax = plt.subplots(figsize=(6, 2)) # Increased height slightly for 2-line title
    for micro_id in range(mbn):
        attn_end = start_time + attn_time
        dispatch_end = max(attn_end, attn_tmp) + dispatch_time
        ffn_end = max(dispatch_end, dispatch_tmp) + moe_time
        combine_end = max(ffn_end, ffn_tmp) + combine_time

        tasks = [
            ("attn", start_time, attn_end),
            ("dispatch", max(attn_end, attn_tmp), dispatch_end),
            ("moe", max(dispatch_end, dispatch_tmp), ffn_end),
            ("combine", max(ffn_end, ffn_tmp), combine_end)
        ]
        attn_tmp, dispatch_tmp, ffn_tmp = dispatch_end, ffn_end, combine_end
        start_time = attn_end  

        for i, (label, start, end) in enumerate(tasks):
            ax.barh(len(tasks) - i - 1, end - start, left=start, height=1, align='center', edgecolor='black', color=color_map[label])
            ax.text(start + (end - start) / 2, len(tasks) - i - 1, str(int(end-start)) + 'us', ha='center', va='center')

    ax.set_yticks([len(tasks) - i - 1 for i in range(len(tasks))])
    ax.set_yticklabels([task[0] for task in tasks])
    
    # Format title with additional info (multi-line)
    # Line 1: TPOT, KV Len, Total Die
    line1 = f'TPOT: {tpot}ms, KV Len: {kv_len}, Total Die: {total_die}'
    
    # Line 2: E2E Time, Throughput, Attn BS
    # Assuming e2e_time is in seconds (as per previous notebook experience), convert to ms for display if needed
    # But usually e2e_time from raw CSV might be small seconds. Let's display as is with 's' or convert to ms.
    # Previous context in notebook: user asked to convert to ms.
    # Let's assume input is seconds and display in ms for consistency with "TPOT (ms)".
    # Or just display raw value if unsure. Let's try to infer or just label it.
    # I'll label it "E2E: {val*1000:.2f}ms" assuming it is seconds.
    
    e2e_ms = e2e_time
    line2_parts = []
    line2_parts.append(f'E2E: {e2e_ms:.2f}ms')
    line2_parts.append(f'Throughput: {throughput:.1f}')
    line2_parts.append(f'Attn BS: {attn_bs:.1f}')

    if mbn >= 2 and attn_die is not None and ffn_die is not None:
         line2_parts.append(f'Attn Die: {int(attn_die)}')
         line2_parts.append(f'FFN Die: {int(ffn_die)}')
         
    title = line1 + '\n' + ", ".join(line2_parts)
    ax.set_title(title, fontsize=10)
    
    # Organize output by model / tpot / kv_len
    # Path: data/images/pipeline/{model_name}/tpot{tpot}_kv{kv_len}/mbn{mbn}_die{total_die}.png
    save_dir = os.path.join('data/images/pipeline', model_name, f'tpot{tpot}_kv{kv_len}')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f'mbn{mbn}_die{total_die}.png')
    
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig) # Close to free memory
    print(f"Saved: {save_path}")

def process_directory(directory, mbn):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # Regex to match filename and extract params
    # Filename: ASCENDA3_Pod-{model_name}-tpot{tpot}-kv_len{kv_len}.csv
    pattern = re.compile(r'ASCENDA3_Pod-(.+)-tpot(\d+)-kv_len(\d+)\.csv')
    
    files = os.listdir(directory)
    for f in files:
        match = pattern.match(f)
        if match:
            model_name = match.group(1)
            tpot = int(match.group(2))
            kv_len = int(match.group(3))
            
            full_path = os.path.join(directory, f)
            try:
                df = pd.read_csv(full_path)
                for index, row in df.iterrows():
                    # Extract necessary columns
                    if 'attn_time' in row and 'dispatch_time' in row and 'moe_time' in row and 'combine_time' in row and 'total_die' in row:
                        total_die = int(row['total_die'])
                        
                        # Filter by target dies
                        if total_die not in TARGET_DIES:
                            continue

                        attn_time = row['attn_time']
                        dispatch_time = row['dispatch_time']
                        moe_time = row['moe_time']
                        combine_time = row['combine_time']
                        
                        # New fields
                        attn_bs = row.get('attn_bs', 0)
                        throughput = row.get('throughput', 0)
                        e2e_time = row.get('e2e_time', 0)
                        
                        attn_die = None
                        ffn_die = None
                        if mbn >= 2:
                            attn_die = row.get('attn_die')
                            ffn_die = row.get('ffn_die')
                        
                        create_gantt_chart(
                            mbn=mbn, 
                            attn_time=attn_time, 
                            dispatch_time=dispatch_time, 
                            moe_time=moe_time, 
                            combine_time=combine_time, 
                            total_die=total_die,
                            model_name=model_name,
                            tpot=tpot,
                            kv_len=kv_len,
                            attn_bs=attn_bs,
                            throughput=throughput,
                            e2e_time=e2e_time,
                            attn_die=attn_die,
                            ffn_die=ffn_die
                        )
            except Exception as e:
                print(f"Error processing {f}: {e}")

import concurrent.futures

def main():
    import concurrent.futures

    # Define directories and associated MBN
    dirs_to_process = [
        ('data/deepep', 1),
        ('data/afd/mbn2/best', 2),
        ('data/afd/mbn3/best', 3)
    ]
    
    print("Starting pipeline visualization generation...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_dir = {
            executor.submit(process_directory, directory, mbn): (directory, mbn)
            for directory, mbn in dirs_to_process
        }
        for future in concurrent.futures.as_completed(future_to_dir):
            directory, mbn = future_to_dir[future]
            try:
                future.result()
                print(f"Finished processing MBN={mbn} from {directory}")
            except Exception as exc:
                print(f"Exception while processing {directory} (MBN={mbn}): {exc}")
    print("Done.")

if __name__ == "__main__":
    main()
