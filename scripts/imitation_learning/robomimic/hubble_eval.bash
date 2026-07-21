conda activate isaaclab_robomimic_3_0
cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena
shopt -s globstar

for i in {0..39}; do
    python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --policy_type robomimic_stand \
        --robomimic_checkpoint submodules/IsaacLab/logs/hubble/hubble_${i}/diffusion_policy_hubble_pick_and_place/**/models/model_epoch_600.pth \
        --num_episodes 50 \
        --num_envs 5 \
        --enable_cameras \
        --output_base_dir submodules/IsaacLab/logs/hubble/hubble_${i}/eval_600 \
        hubble_g1_static_pick_and_place
    python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --policy_type robomimic_stand \
        --robomimic_checkpoint submodules/IsaacLab/logs/hubble/hubble_${i}/diffusion_policy_hubble_pick_and_place/**/last.pth \
        --num_episodes 50 \
        --num_envs 5 \
        --enable_cameras \
        --output_base_dir submodules/IsaacLab/logs/hubble/hubble_${i}/eval_last \
        hubble_g1_static_pick_and_place
    echo "--------------------------------"
    echo "Completed evaluation for dp_${i}"
    echo "--------------------------------"
done
