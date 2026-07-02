conda activate isaac_auto_data
cd ~/Documents/Isaac-AutoData/submodules/IsaacLab-Arena/submodules/IsaacLab
mkdir -p logs/long_weekend

for i in {0..5}; do
    case $i in
        3) dataset=../../datasets/teleop/hubble_generated_1000_preproc.hdf5 ;;
        4) dataset=../../datasets/teleop/hubble_horizontal_grasp_generated_preproc.hdf5 ;;
        5) dataset=../../datasets/teleop/hubble_extra_cam_generated_preproc.hdf5 ;;
        *) dataset=../../datasets/teleop/hubble_generated_new_preproc.hdf5 ;;
    esac

    ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
        --config ./scripts/imitation_learning/robomimic/long_weekend/dp_1${i}.json \
        --log_dir long_weekend \
        --task long_weekend_1${i} \
        --algo diffusion_policy \
        --dataset ${dataset}
done

# 10: add joint pos to obs
# 11: add actions to obs
# 12: add both joint pos and actions to obs
# 13: train from 1000 demos
# 14: train from new dataset
# 15: use extra camera
