conda activate isaaclab_robomimic
cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena/submodules/IsaacLab
mkdir -p logs/hubble

for i in {0..39}; do
    ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
        --config ./scripts/imitation_learning/robomimic/configs/dp_${i}.json \
        --log_dir hubble \
        --task hubble_${i} \
        --algo diffusion_policy \
        --dataset ../../datasets/teleop/hubble_generated_new.hdf5
done
