import os 
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter, FileWriter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



tf_files = [] # -> list of paths from writer.log_dir to all files in that directory

for root, dirs, files in os.walk("results"):
    for file_name in files:
        if file_name.startswith("events.out.tfevents."):
            tf_files.append(os.path.join(root,file_name)) # go over every file recursively in the directory

for file_id, file_name in enumerate(tf_files):
    path = os.path.split(file_name)[0] # determine path to folder in which file lies
    

    event_acc = EventAccumulator(file_name)
    event_acc.Reload()
    
    writer = SummaryWriter(log_dir=path)
    # writer = FileWriter(log_dir=path)
    for tag in sorted(event_acc.Tags()["scalars"]):
        if not tag in ["Train/Batch-crossentropyloss", "Val/Batch-crossentropyloss"]:

            for scalar_event in event_acc.Scalars(tag):
                writer.add_scalar(tag,scalar_event.value,scalar_event.step, walltime=scalar_event.wall_time)
                # writer.add_event(scalar_event, walltime=scalar_event.wall_time)
    writer.close()
    os.remove(file_name)
    # exit()