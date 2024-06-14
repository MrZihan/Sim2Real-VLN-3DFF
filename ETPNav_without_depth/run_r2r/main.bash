export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_r2r
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      IL.iters 15000
      IL.lr 1e-5
      IL.log_every 500
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 3000
      IL.load_from_ckpt True
      IL.is_requeue True
      IL.waypoint_aug  True
	  IL.ckpt_to_load data/logs/checkpoints/release_r2r/ckpt.iter10000.pth
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path pretrained/model_step_100000.pt
      "

flag2=" --exp_name release_r2r
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1]
      TORCH_GPU_IDS [0,1]
      GPU_NUMBERS 2
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
	  MODEL.pretrained_path pretrained/model_step_100000.pt
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter15000.pth
      IL.back_algo control
      "

flag3="--exp_name release_r2r
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1]
      TORCH_GPU_IDS [0,1]
      GPU_NUMBERS 2
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 2445 run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      python3 -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run.py $flag3
      ;;
esac