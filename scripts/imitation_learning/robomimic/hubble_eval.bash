conda activate isaaclab_robomimic_clone
cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena
shopt -s globstar

for i in {0..39}; do
    python isaaclab_arena/evaluation/policy_runner.py \
        --viz kit \
        --policy_type robomimic_stand \
        --robomimic_checkpoint submodules/IsaacLab/logs/hubble/hubble_${i}/diffusion_policy_hubble_pick_and_place/**/models/model_epoch_600.pth \
        --num_episodes 10 \
        --enable_cameras \
        hubble_g1_static_pick_and_place
    echo "--------------------------------"
    echo "Completed evaluation for dp_${i}"
    echo "--------------------------------"
done
