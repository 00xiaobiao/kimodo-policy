
source ~/miniconda3/bin/activate twist2
# task_name="0819_shelf"

cd deploy_real

#robot_ip="10.42.0.35"
robot_ip="127.0.0.1"
# robot_ip="192.168.110.24"
data_frequency=30
save_folder='/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data'


python server_data_record_with_third_smplx_qpos.py --data_folder ${save_folder} --frequency ${data_frequency} --robot_ip ${robot_ip}
        