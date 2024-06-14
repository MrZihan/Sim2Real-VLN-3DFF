export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_r2r
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1]
      TORCH_GPU_IDS [0,1]
      GPU_NUMBERS 2
      NUM_ENVIRONMENTS 2
      IL.iters 20000
      IL.lr 1e-4
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 2000
      IL.load_from_ckpt False
      IL.is_requeue False
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path data/ckpt.iter12000.pth
      "

flag2=" --exp_name release_r2r
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/r2r/checkpoints/release_r2r/ckpt.iter17400.pth
	  MODEL.pretrained_path data/ckpt.iter12000.pth
      IL.back_algo control
      "
# data/logs/r2r/checkpoints/release_r2r/ckpt.iter17400.pth

flag3="--exp_name release_r2r
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/r2r/checkpoints/release_r2r/ckpt.iter20000.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "


#CUDA_VISIBLE_DEVICES='6,7' python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 2334 run.py $flag1

CUDA_VISIBLE_DEVICES='5' python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 2335 run.py $flag2

#CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag3
