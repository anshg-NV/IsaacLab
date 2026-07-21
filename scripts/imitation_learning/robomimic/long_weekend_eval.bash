conda activate isaaclab_robomimic_3_0
cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena
shopt -s globstar

read -p "HEY DID YOU REVERT THE G1.PY CAMERA SIZE??? (y/n) " answer
if [ "$answer" != "y" ]; then
    exit 1
fi
for i in 00 01 02 03 04 05 10 11 12 13 14 15; do
    python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --policy_type robomimic_stand \
        --robomimic_checkpoint submodules/IsaacLab/logs/long_weekend/long_weekend_${i}/diffusion_policy_hubble_pick_and_place/**/models/model_epoch_600.pth \
        --num_episodes 50 \
        --num_envs 5 \
        --enable_cameras \
        --output_base_dir submodules/IsaacLab/logs/long_weekend/long_weekend_${i}/eval_600 \
        hubble_g1_static_pick_and_place
    python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --policy_type robomimic_stand \
        --robomimic_checkpoint submodules/IsaacLab/logs/long_weekend/long_weekend_${i}/diffusion_policy_hubble_pick_and_place/**/last.pth \
        --num_episodes 50 \
        --num_envs 5 \
        --enable_cameras \
        --output_base_dir submodules/IsaacLab/logs/long_weekend/long_weekend_${i}/eval_last \
        hubble_g1_static_pick_and_place
    echo "--------------------------------"
    echo "Completed evaluation for long_weekend_${i}"
    echo "--------------------------------"
done
