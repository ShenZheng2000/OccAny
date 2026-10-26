# usage: bash sh/setup_eval_env.sh

# ---- paths ----
export PROJECT=/data3/shenzhen/World_Projects/OccAny
export SCRATCH=/scratch/shenzhen/Datasets/occany_eval

# ---- datasets ----
export KITTI_ROOT=/scratch/shenzhen/Datasets/KITTI          # contains dataset/
export NUSCENES_ROOT=/scratch/shenzhen/Datasets/nuscenes    # contains annotations.json, gts/, imgs/

# ---- shard (same for boxes + eval) ----
# debug: every 500 val samples. For full val set WORLD=1.
export WORLD=1
export PID=0

# ---- GPU ----
export CUDA_VISIBLE_DEVICES=6

# ---- output dirs ----
mkdir -p "$SCRATCH"/{ssc_voxel_pred,ssc_output,data/kitti_processed,data/nuscenes_processed}

cd $PROJECT

# ---- step 0: build GroundingDINO C++ ops (only if not built) ----
# pip install -e third_party/Grounded-SAM-2
# pip install --no-build-isolation -e third_party/Grounded-SAM-2/grounding_dino

# ---- step 1: GroundingDINO boxes for nuScenes -> $SCRATCH/data/nuscenes_processed ----
python extract_gdino_boxes_nuscenes.py --image_size 1328 --box_threshold 0.05 --text_threshold 0.05 --world $WORLD --pid $PID --silent

# # ---- step 2: eval, nuScenes surround geometry -> $SCRATCH/ssc_voxel_pred ----
SLURM_ARRAY_TASK_ID=$PID EXP_LIST=occany EXP_ID=3 bash sh/eval_occany.sh

# # ---- step 3: metrics ----
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=3 bash sh/compute_metric.sh
