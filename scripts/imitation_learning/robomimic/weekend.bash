# conda activate isaaclab_robomimic
# cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena/submodules/IsaacLab
# mkdir -p logs/weekend

# for i in {0..10}; do
#     ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
#         --config ./source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/agents/robomimic/dp_${i}.json \
#         --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
#         --algo diffusion_policy \
#         --dataset ./datasets/generated_dataset_visuomotor.hdf5
#     mv logs/robomimic logs/weekend/robomimic_${i}
# done

conda activate isaaclab_robomimic
cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena/submodules/IsaacLab
shopt -s globstar

for i in {0..10}; do
    if (( i <= 8 )); then
        epochs=600
    elif (( i == 9 )); then
        epochs=1200
    else
        epochs=300
    fi

    ./isaaclab.sh -p scripts/imitation_learning/robomimic/play_05.py \
        --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
        --checkpoint logs/weekend/robomimic_${i}/**/models/model_epoch_${epochs}.pth \
        --enable_cameras --viz kit
    echo "--------------------------------"
    echo "Completed training for dp_${i}"
    echo "--------------------------------"
done
