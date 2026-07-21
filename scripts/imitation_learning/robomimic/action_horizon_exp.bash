conda activate isaaclab_robomimic_3_0
cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena
shopt -s globstar

for i in 8; do
    python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --policy_type robomimic_stand \
        --robomimic_checkpoint submodules/IsaacLab/logs/hubble/action_horizon_exp/golden_${i}.pth \
        --num_episodes 50 \
        --num_envs 5 \
        --enable_cameras \
        --output_base_dir submodules/IsaacLab/logs/hubble/action_horizon_exp/golden_${i} \
        hubble_g1_static_pick_and_place
    echo "--------------------------------"
    echo "Completed evaluation for dp_${i}"
    echo "--------------------------------"
done
