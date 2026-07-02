conda activate isaac_auto_data
cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena/submodules/IsaacLab
mkdir -p logs/long_weekend

for i in {0..5}; do
    ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
        --config ./scripts/imitation_learning/robomimic/long_weekend/dp_0${i}.json \
        --log_dir long_weekend \
        --task long_weekend_0${i} \
        --algo diffusion_policy \
        --dataset ../../datasets/teleop/hubble_generated_new_preproc.hdf5
done

# 00: control
# 01: diffusion steps = 5
# 02: diffusion steps = 3
# 03: action horizon = 4
# 04: prediction horizon = 24
# 05: prediction horizon = 32
