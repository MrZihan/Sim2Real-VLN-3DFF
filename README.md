## Sim-to-Real Transfer via 3D Feature Fields for Vision-and-Language Navigation

#### Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu and Shuqiang Jiang

>Vision-and-language navigation (VLN) enables the agent to navigate to a remote location in 3D environments following the natural language instruction. In this field, the agent is usually trained and evaluated in the navigation simulators, lacking effective approaches for sim-to-real transfer. The VLN agents with only a monocular camera exhibit extremely limited performance, while the mainstream VLN models trained with panoramic observation, perform better but are difficult to deploy on most monocular robots. For this case, we propose a sim-to-real transfer approach to endow the monocular robots with panoramic traversability perception and panoramic semantic understanding, thus smoothly transferring the high-performance panoramic VLN models to the common monocular robots. In this work, the semantic traversable map is proposed to predict agent-centric navigable waypoints, and the novel view representations of these navigable waypoints are predicted through the 3D feature fields. These methods broaden the limited field of view of the monocular robots and significantly improve navigation performance in the real world. Our VLN system outperforms previous SOTA monocular VLN methods in R2R-CE and RxR-CE benchmarks within the simulation environments and is also validated in real-world environments, providing a practical and high-performance solution for real-world VLN.

<div align=center><img src="https://github.com/MrZihan/Sim2Real-VLN-3DFF/blob/main/Figure/Figure1.jpg" width="700px" alt="Figure 1. The VLN models equipped with a monocular camera have limited navigation success rates of less than 39% on the R2R-CE Val Unseen split. Most VLN models are trained and evaluated in the simulator [6] with the panoramic observation, achieving navigation success rates of over 57%, but hard to deploy on real-world robots."/></div>

<div align=center><img src="https://github.com/MrZihan/Sim2Real-VLN-3DFF/blob/main/Figure/Figure2.jpg" width="700px" alt="Figure 2. The sim-to-real transfer framework via semantic traversable map and 3D feature fields for vision-and-language navigation."/></div>

### Requirements

1. Install `Habitat simulator`: follow instructions from [ETPNav](https://github.com/MarSaKi/ETPNav) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE).
2. (Optional) Download [MP3D Scene Semantic Pclouds](https://drive.google.com/file/d/1u4SKEYs4L5RnyXrIX-faXGU1jc16CTkJ/view) for pre-training the semantic and occupancy map predictor, following [CM2](https://github.com/ggeorgak11/CM2).
3. (Optional) Download [GT annotation of waypoints](https://drive.google.com/drive/folders/1wpuGAO-rRalPKt8m1-QIvlb_Pv1rYJ4x?usp=sharing) for pre-training the traversable map predictor, following [CWP](https://github.com/wz0919/waypoint-predictor).
4. Install `torch_kdtree` for K-nearest feature search from [torch_kdtree](https://github.com/thomgrand/torch_kdtree), following [HNR-VLN](https://github.com/MrZihan/HNR-VLN).
5. Install `tinycudann` for faster multi-layer perceptrons (MLPs) from [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), following [HNR-VLN](https://github.com/MrZihan/HNR-VLN).
6. Download the preprocessed data and checkpoints from [BaiduNetDisk](https://pan.baidu.com/s/1RL9VI5NU9uTXLOyymhmx1w?pwd=ugi2).

### (Optional) Pre-train the Semantic Traversable Map
```
cd Traversable_Map
bash run_r2r/main.bash train 2341
```

### (Optional) Pre-train the 3D Feature Fields

Follow the [HNR-VLN](https://github.com/MrZihan/HNR-VLN) and use the [CLIP-ViT-B/16](https://github.com/openai/CLIP/blob/main/clip/clip.py) as the visual encoder.

### (Optional) Pre-train the ETPNav without depth feature
Follow [ETPNav](https://github.com/MarSaKi/ETPNav), download the pretraining datasets [link](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0) (the same one used in [DUET](https://github.com/cshizhe/VLN-DUET)) and precomputed features [link](https://drive.google.com/file/d/1D3Gd9jqRfF-NjlxDAQG_qwxTIakZlrWd/view?usp=sharing), unzip in folder `pretrain_src`

```
cd ETPNav_without_depth
bash pretrain_src/run_pt/run_r2r.bash 2342
```

### (Optional) Finetune the ETPNav without depth feature
Follow [ETPNav](https://github.com/MarSaKi/ETPNav), for R2R-CE
```
cd ETPNav_without_depth
bash run_r2r/main.bash train 2343
```

Follow [ETPNav](https://github.com/MarSaKi/ETPNav), for RxR-CE
```
cd ETPNav_without_depth
bash run_rxr/main.bash train 2343
```

### Train and evaluate the monocular ETPNav with 3D Feature Fields
```
cd VLN_3DFF
bash run_r2r/main.bash train 2344 # training
bash run_r2r/main.bash eval 2344 # evaluation
bash run_r2r/main.bash inter 2344 # inference
```
```
cd VLN_3DFF
bash run_rxr/main.bash train 2344  # training
bash run_rxr/main.bash eval  2344  # evaluation
bash run_rxr/main.bash inter 2344  # inference
```

### (Optional) Run in Interbotix LoCoBot WX250 for real-world VLN
Ensure the robot and the server are on the same local area network (LAN).

Fill in the `Server_IP` and `Robot_IP` correctly in Server_Code/run.py and Robot_Code/robot.py.

Run the VLN model in the server:
```
cd Server_Code
python3 run.py
```

Run the control code in the robot:
```
cd Robot_Code
python3 run.py
```
