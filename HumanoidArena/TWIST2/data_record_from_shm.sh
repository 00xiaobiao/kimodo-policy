#!/bin/bash

# Data recording script that reads directly from IsaacLab shared memory
# No network connection needed!
# Display window is always enabled

source ~/miniconda3/bin/activate twist2

cd deploy_real

data_frequency=30
save_folder="/media/yixiao/Extreme_Pro/weisheng"

# No robot_ip needed - reads directly from shared memory!
python server_data_record_from_shm.py \
    --data_folder ${save_folder} \
    --frequency ${data_frequency}
