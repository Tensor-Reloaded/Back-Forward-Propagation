import os
import sys
import subprocess
import shutil
import glob
import re

# for days in os.listdir("./outputs"):
#     if os.path.isdir(os.path.join("./outputs/", days)):
#         for hours in os.listdir(os.path.join('./outputs',days)):
#             shutil.copytree(os.path.join('./outputs',days,hours, "runs"), "./results/", dirs_exist_ok=True)

def rename(f):
    new_name = re.sub(r'events.out.tfevents.*', 'events.out.tfevents.123456.Author1-Author2.0', f)
    # if new_name != f:
    #     os.remove(f)
    if not os.path.isfile(new_name):
        os.rename(f, new_name)
    
for f in glob.glob('./results/**/events.out.tfevents.*', recursive=True):
    rename(f)
# for f in glob.glob('./results_dba_adam/**/events.out.tfevents.*', recursive=True):
#     rename(f)
# for f in glob.glob('./results_dba_sgd/**/events.out.tfevents.*', recursive=True):
#     rename(f)
# for f in glob.glob('./results_exploratory/**/events.out.tfevents.*', recursive=True):
#     rename(f)