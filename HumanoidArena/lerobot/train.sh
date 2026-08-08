PYTHONPATH=src python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=local/sonic-football-0411-64-40 \
  --dataset.root=/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/HumanoidArena_datasets/HOI_football/0411_sonic_smpl_pose6d_aligned \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.device=cuda \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.optimizer_lr=1e-5 \
  --policy.push_to_hub=false \
  --batch_size=80 \
  --steps=50000 \
  --output_dir=/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/train/act_sonic_football_rand_0413_64_40 \
  --job_name=act_sonic_football_rand_0413_64_40


# PYTHONPATH=src python src/lerobot/scripts/lerobot_train.py \
#   --dataset.repo_id=local/sonic-smpl-pose6d-vla \
#   --dataset.root=/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/HumanoidArena_datasets/HOI_football/0409_sonic_smpl_pose6d \
#   --dataset.image_transforms.enable=true \
#   --policy.type=act \
#   --policy.device=cuda \
#   --policy.chunk_size=20 \
#   --policy.n_action_steps=20 \
#   --policy.optimizer_lr=1e-5 \
#   --policy.push_to_hub=false \
#   --batch_size=80 \
#   --steps=50000 \
#   --output_dir=/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/train/act_sonic_football_rand_0409 \
#   --job_name=act_sonic_football_rand_0409


PYTHONPATH=src python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=local/sonic-football-0411-64-40 \
  --dataset.root=/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/HumanoidArena_datasets/HOI_football/0411_sonic_smpl_pose6d_aligned \
  --dataset.image_transforms.enable=true \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.n_obs_steps=2 \
  --policy.horizon=24 \
  --policy.n_action_steps=20 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --steps=50000 \
  --output_dir=/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/train/diffusion_sonic_football_rand_0413_64_40 \
  --job_name=diffusion_sonic_football_rand_0413_64_40
