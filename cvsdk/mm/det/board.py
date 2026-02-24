import click
import pandas as pd
import json
import os
import time
import subprocess
import signal
from torch.utils.tensorboard import SummaryWriter

def parse_json_log_file(log_file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch_logs = []
    eval_logs = []

    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if 'iter' in log_entry and 'epoch' in log_entry:
                    batch_logs.append(log_entry)
                elif any(k.startswith("coco/") for k in log_entry):
                    eval_logs.append(log_entry)
            except json.JSONDecodeError:
                print("Warning: Skipping invalid JSON line.")
                print(line)
                continue

    batch_df = pd.DataFrame(batch_logs)
    eval_df = pd.DataFrame(eval_logs)
    return batch_df, eval_df

def log_metrics_to_tensorboard(batch_df: pd.DataFrame, eval_df: pd.DataFrame, log_dir: str="./runs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    for _, row in batch_df.iterrows():
        step = row["step"]
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in ("step", "epoch", "iter", "memory"):
                writer.add_scalar(f"train/{key}", value, global_step=step)

    for _, row in eval_df.iterrows():
        step = row["step"]
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "step":
                writer.add_scalar(f"eval/{key}", value, global_step=step)

    writer.close()