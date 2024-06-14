import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss
from vlnce_baselines.models.utils import get_angle_fts, calculate_vp_rel_pos_fts

import vlnce_baselines.waypoint_networks.utils as utils
from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele, load_gt_navigability, get_gt_nav_map
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from vlnce_baselines.waypoint_networks.semantic_grid import SemanticGrid
from vlnce_baselines.waypoint_networks import get_img_segmentor_from_options
from vlnce_baselines.waypoint_networks.resnetUnet import ResNetUNet
import vlnce_baselines.waypoint_networks.viz_utils as viz_utils

from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
import habitat_sim
import cv2
from PIL import Image
#import clip
import open3d as o3d


@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.config = config
        self.batch_size = self.config.IL.batch_size


    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            #H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    #camera_config.WIDTH = H
                    #camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()
        
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )

        env_num = self.envs.num_envs
        self.batch_size = env_num
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )


        return observation_space, action_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )

        ''' initialize the waypoint predictor here '''
        #####################
        self.navigability_map = load_gt_navigability("data/habitat_connectivity_graph/train_with_pos.json")
        self.waypoint_id_to_position = json.load(open("data/habitat_connectivity_graph/train.json",'r')) # self.waypoint_id_to_position[scene_id]['nodes'][waypoint_id]

        self.eval_navigability_map = load_gt_navigability("data/habitat_connectivity_graph/val_unseen_with_pos.json")
        self.eval_waypoint_id_to_position = json.load(open("data/habitat_connectivity_graph/val_unseen.json",'r')) # self.waypoint_id_to_position[scene_id]['nodes'][waypoint_id]
        self.navigability_map.update(self.eval_navigability_map)
        self.waypoint_id_to_position.update(self.eval_waypoint_id_to_position)


        self.position_to_waypoint_id = {}
        for scene_id in self.navigability_map:
            self.position_to_waypoint_id[scene_id] = {}
            for waypoint_id in self.navigability_map[scene_id]:
                position = self.navigability_map[scene_id][waypoint_id]['source_pos']
                self.position_to_waypoint_id[scene_id][str(position)] = waypoint_id

        ####################

        n_object_classes = 27

        ## Load the pre-trained img segmentation model
        self.img_segmentor = get_img_segmentor_from_options(n_object_classes,1.0)
        self.img_segmentor = self.img_segmentor.to(self.device)

        if self.config.GPU_NUMBERS > 1:
            self.img_segmentor = DDP(self.img_segmentor,device_ids=[self.device], output_device=self.device)
        else:
            self.img_segmentor = torch.nn.DataParallel(self.img_segmentor)

        checkpoint = torch.load("pretrained_models/segm.pt")
        self.img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'])         
        self.img_segmentor.eval()

        self.policy.net.occupancy_map_predictor = ResNetUNet(3,3,True)
        self.policy.net.semantic_map_predictor = ResNetUNet(n_object_classes+3,n_object_classes,True)
        self.policy.net.waypoint_predictor = ResNetUNet(n_object_classes+3,1,True)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()

        #self.noise_filter = torch.nn.Conv2d(1, 1, (5, 5), padding=(2,2)).to(self.device)
        #noise_filter_weight = torch.ones(1,1,5,5).to(self.device) / (5.*5.)
        #self.noise_filter.weight = torch.nn.Parameter(noise_filter_weight)
        #self.noise_filter.eval()

        # get point cloud and labels of scene
        self.preprocessed_scenes_dir = "data/scene_datasets/mp3d_scene_pclouds/"
        object_labels = list(range(n_object_classes))

        self.pclouds = {}
        self.color_pcloud = {}
        self.label_seq_spatial = {}
        self.label_seq_objects = {}
        '''
        for scene_id in self.navigability_map:
            xyz, label_seq_spatial, label_seq_objects= utils.load_scene_pcloud(self.preprocessed_scenes_dir, scene_id, n_object_classes)
            self.color_pcloud[scene_id] = utils.load_scene_color(self.preprocessed_scenes_dir, scene_id)
            self.pclouds[scene_id] = xyz
            self.label_seq_spatial[scene_id] = label_seq_spatial
            self.label_seq_objects[scene_id] = label_seq_objects

        '''

        self.img_segm_size = (128,128)
        ## Build necessary info for ground-projecting the semantic segmentation
        self._xs, self._ys = torch.tensor(np.array(np.meshgrid(np.linspace(-1,1,self.img_segm_size[0]), np.linspace(1,-1,self.img_segm_size[1]))), device=self.device)
        self._xs = self._xs.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        self._ys = self._ys.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        _x, _y = torch.tensor(np.array(np.meshgrid(np.linspace(0, self.img_segm_size[0]-1, self.img_segm_size[0]), 
                                                    np.linspace(0, self.img_segm_size[1]-1, self.img_segm_size[1]))), device=self.device)
        _xy_img = torch.cat((_x.reshape(1,self.img_segm_size[0],self.img_segm_size[1]), _y.reshape(1,self.img_segm_size[0],self.img_segm_size[1])), dim=0)
        _points2D_step = _xy_img.reshape(2, -1)
        self._points2D_step = torch.transpose(_points2D_step, 0, 1) # Npoints x 2  


        self.policy.net = self.policy.net.to(self.device)
        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net, device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=self.config.IL.lr)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
                self.policy.net = self.policy.net.module
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
            if config.IL.is_requeue:
                try:
                    self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                except:
                    print("Optim_state is not loaded")

            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()

        #self.waypoint_predictor.eval()


        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)


        #####################
        self.navigability_map = load_gt_navigability("data/habitat_connectivity_graph/train_with_pos.json")
        self.eval_navigability_map = load_gt_navigability("data/habitat_connectivity_graph/val_unseen_with_pos.json")
        self.navigability_map.update(self.eval_navigability_map)
        self.position_to_waypoint_id = {}
        for scene_id in self.navigability_map:
            self.position_to_waypoint_id[scene_id] = {}
            for waypoint_id in self.navigability_map[scene_id]:
                position = self.navigability_map[scene_id][waypoint_id]['source_pos']
                self.position_to_waypoint_id[scene_id][str(position)] = waypoint_id

        ####################

        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            #if self.loss != 0.:
            #    self.loss.backward()
            #    torch.nn.utils.clip_grad_norm_(parameters=self.policy.parameters(), max_norm=5, norm_type=2)
            #    self.optimizer.step()

            print(self.loss)

            self.scaler.scale(self.loss).backward() # self.loss.backward()
            self.scaler.step(self.optimizer)        # self.optimizer.step()
            self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
           
        gc.collect()
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            #H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    #camera_config.WIDTH = H
                    #camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        env_num = self.envs.num_envs
        self.batch_size = env_num
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        #self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def RGB_to_BGR(self, cvimg):
        pilimg = cvimg.copy()
        pilimg[:, :, 0] = cvimg[:, :, 2]
        pilimg[:, :, 2] = cvimg[:, :, 0]
        return pilimg

                    


    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0


        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)


        loss = 0.
        total_actions = 0.
     

        scene_ids = [item.scene_id.split('/')[-2] for item in self.envs.current_episodes()]
        waypoint_ids = [random.choice(list(self.navigability_map[scene_id].keys())) for scene_id in scene_ids]
       

        headings = [random.uniform(-math.pi,math.pi) for b in range(self.batch_size)]
        visited_waypoints = [[waypoint_ids[b]] for b in range(self.batch_size)]

        positions = []
        observations = [None for b in range(self.batch_size)]
        for b in range(self.batch_size):
            positions.append(self.waypoint_id_to_position[scene_ids[b]]['nodes'][waypoint_ids[b]])

            q1 = math.cos(headings[b]/2)
            q2 = math.sin(headings[b]/2)

            rotation = np.quaternion(q1,0,q2,0)
            camera_obs = self.envs.call_at(b, "get_observation",{"source_position":positions[b],"source_rotation":rotation,"keep_agent_at_new_pose":True})

            observations[b] = camera_obs

        next_pos = [None for b in range(self.batch_size)]



        hfov = 90. * np.pi / 180.
        vfov = 90. * np.pi / 180.
        map_config={'hfov':hfov,'vfov':vfov,'global_dim':(512,512),'grid_dim':(192,192),'heatmap_size':192,'cell_size':0.05,'img_segm_size':(128,128),'spatial_labels':3,'object_labels':27,'img_size':[256,256],'occupancy_height_thresh':-1.0,'norm_depth':True}
        # 3d info
        xs, ys = torch.tensor(np.array(np.meshgrid(np.linspace(-1,1,map_config['img_size'][0]), np.linspace(1,-1,map_config['img_size'][1]))), device='cuda')

        xs = xs.reshape(1,map_config['img_size'][0],map_config['img_size'][1])
        ys = ys.reshape(1,map_config['img_size'][0],map_config['img_size'][1])
        K = np.array([
            [1 / np.tan(map_config['hfov'] / 2.), 0., 0., 0.],
            [0., 1 / np.tan(map_config['vfov'] / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        inv_K = torch.tensor(np.linalg.inv(K), device=self.device)



        ##########
        for scene_id in scene_ids:
            if scene_id not in self.pclouds:
                xyz, label_seq_spatial, label_seq_objects= utils.load_scene_pcloud(self.preprocessed_scenes_dir, scene_id, map_config['object_labels'])
                self.color_pcloud[scene_id] = utils.load_scene_color(self.preprocessed_scenes_dir, scene_id)
                self.pclouds[scene_id] = xyz
                self.label_seq_spatial[scene_id] = label_seq_spatial
                self.label_seq_objects[scene_id] = label_seq_objects
        ############

        # For each episode we need a new instance of a fresh global grid
        sg_map = SemanticGrid(self.batch_size, map_config['grid_dim'], map_config['heatmap_size'], map_config['cell_size'],
                            spatial_labels=map_config['spatial_labels'], object_labels=map_config['object_labels'])

        sg_map_global = SemanticGrid(self.batch_size, map_config['global_dim'], map_config['heatmap_size'], map_config['cell_size'],
                            spatial_labels=map_config['spatial_labels'], object_labels=map_config['object_labels'])

        abs_poses = [[] for b in range(self.batch_size)]

        for stepk in range(20):
            total_actions += self.envs.num_envs
           
            if mode == 'train':

                # agent's current position and heading
                for ob_i in range(len(observations)):
                    agent_state_i = self.envs.call_at(ob_i,
                            "get_agent_info", {})
                    positions[ob_i] = agent_state_i['position']
                    headings[ob_i] = agent_state_i['heading']

                batch_img = []
                batch_depth = []
                batch_local3D_step = []
                batch_rel_abs_pose = []
                batch_candidate_rel_pos = []
                batch_gt_map_semantic = []
                batch_gt_map_occupancy = []
                _, candidate_pos = get_gt_nav_map(self.navigability_map,scene_ids,waypoint_ids)

                for b in range(self.batch_size):

                    #####################################

                    img = observations[b]['rgb']
                    depth = observations[b]['depth'].reshape(map_config['img_size'][0], map_config['img_size'][1], 1)

                    viz_img = img
                    img = torch.tensor(img).to(self.device)
                    depth = torch.tensor(depth).to(self.device)

                    if map_config['norm_depth']:
                        depth_abs = utils.unnormalize_depth(depth, min=0.0, max=10.0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    batch_img.append(img.unsqueeze(0))
                    batch_depth.append(depth_abs.unsqueeze(0))

                    local3D_step = utils.depth_to_3D(depth_abs, map_config['img_size'], xs, ys, inv_K)
                    batch_local3D_step.append(local3D_step)

                    agent_state = self.envs.call_at(b,"get_agent_state", {})

                    agent_pose, y_height = utils.get_sim_location(agent_state=agent_state)
                    abs_poses[b].append(agent_pose)

                    scene_id = scene_ids[b]
                    # get gt map from agent pose for visualization later (pose is at the center looking upwards)
                    x, y, z, label_seq, color_pcloud = utils.slice_scene(x=self.pclouds[scene_id][0].copy(),
                                                                        y=self.pclouds[scene_id][1].copy(),
                                                                        z=self.pclouds[scene_id][2].copy(),
                                                                        label_seq=self.label_seq_objects[scene_id].copy(),
                                                                        height=y_height,
                                                                        color_pcloud=self.color_pcloud[scene_id])

                    gt_map_semantic, gt_map_occupancy, _ = utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[b][stepk],
                                                                grid_dim=map_config['grid_dim'], cell_size=map_config['cell_size'], color_pcloud=color_pcloud, z=z) 
                    batch_gt_map_semantic.append(gt_map_semantic)
                    batch_gt_map_occupancy.append(gt_map_occupancy)

                    try:
                        next_pos[b] = random.choice(candidate_pos[b])
                        waypoint_ids[b] = self.position_to_waypoint_id[scene_ids[b]][str(next_pos[b])]
                        visited_waypoints[b].append(waypoint_ids[b])
                        next_pos[b] = self.waypoint_id_to_position[scene_ids[b]]['nodes'][waypoint_ids[b]]
                        if waypoint_ids[b] in visited_waypoints[b]:
                            next_pos[b] = random.choice(candidate_pos[b])
                            waypoint_ids[b] = self.position_to_waypoint_id[scene_ids[b]][str(next_pos[b])]
                            visited_waypoints[b].append(waypoint_ids[b])
                            next_pos[b] = self.waypoint_id_to_position[scene_ids[b]]['nodes'][waypoint_ids[b]]
                        
                    except:
                        print('Position error! Skip...')
                        for i in range(len(candidate_pos[b])):
                            if str(candidate_pos[b][i]) in self.position_to_waypoint_id[scene_ids[b]]:
                                next_pos[b] = candidate_pos[b][i]
                                waypoint_ids[b] = self.position_to_waypoint_id[scene_ids[b]][str(next_pos[b])]
                                visited_waypoints[b].append(waypoint_ids[b])
                                next_pos[b] = self.waypoint_id_to_position[scene_ids[b]]['nodes'][waypoint_ids[b]]
                                break

                    candidate_rel_pos = []
                    for candidate_pos_item in candidate_pos[b]:
                        candidate_rel_pos_item, candidate_visible = utils.transform_to_map_coords(sg=sg_map, position=candidate_pos_item, abs_pose=abs_poses[b][stepk],grid_size=map_config['grid_dim'][0],cell_size=map_config['cell_size'],device=self.device)
                        candidate_rel_pos.append(candidate_rel_pos_item[0][0].numpy().tolist())

                    batch_candidate_rel_pos.append(candidate_rel_pos)
                    # Keep track of the agent's relative pose from the initial position
                    rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[b][stepk], pos1=abs_poses[b][0])
                    _rel_abs_pose = torch.Tensor(rel_abs_pose).unsqueeze(0).float()
                    _rel_abs_pose = _rel_abs_pose.to(self.device)
                    batch_rel_abs_pose.append(_rel_abs_pose)


                ### Run the img segmentation model to get the ground-projected semantic segmentation
                batch_abs_poses = torch.tensor(abs_poses).to(self.device)
                batch_rel_abs_pose = torch.cat(batch_rel_abs_pose,dim=0)
                batch_gt_map_semantic = torch.cat(batch_gt_map_semantic,dim=0)
                batch_gt_map_occupancy = torch.cat(batch_gt_map_occupancy,dim=0)

                batch_img = torch.cat(batch_img,dim=0)
                batch_depth = torch.cat(batch_depth,dim=0)
                depth_img = batch_depth.clone().permute(0,3,1,2)

                depth_img = F.interpolate(depth_img, size=map_config['img_segm_size'], mode='nearest')
                imgData = utils.preprocess_img(batch_img, cropSize=map_config['img_segm_size'], pixFormat='NCHW', normalize=True)

                segm_batch = {'images':imgData.to(self.device).unsqueeze(1),
                            'depth_imgs':depth_img.to(self.device).unsqueeze(1)}

                pred_ego_sseg, img_segm = utils.run_img_segm(model=self.img_segmentor, 
                                                        input_batch=segm_batch, 
                                                        object_labels=map_config['object_labels'], 
                                                        crop_size=map_config['global_dim'], 
                                                        cell_size=map_config['cell_size'],
                                                        xs=self._xs,
                                                        ys=self._ys,
                                                        inv_K=inv_K,
                                                        points2D_step=self._points2D_step)   


                # do ground-projection, update the projected map
                ego_grid_sseg_3 = utils.est_occ_from_depth(batch_local3D_step, grid_dim=map_config['global_dim'], cell_size=map_config['cell_size'], 
                                                                                device=self.device, occupancy_height_thresh=map_config['occupancy_height_thresh'])

                # Transform the ground projected egocentric grids to geocentric using relative pose
                occup_grid_sseg = sg_map_global.spatialTransformer(grid=ego_grid_sseg_3, pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                semantic_grid_sseg = sg_map_global.spatialTransformer(grid=pred_ego_sseg[:,0], pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                # step_geo_grid contains the map snapshot every time a new observation is added
                step_occup_grid_sseg, step_segm_grid_sseg = sg_map_global.update_proj_grid_bayes(occup_grid_sseg.unsqueeze(1),semantic_grid_sseg.unsqueeze(1))
                # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)

                step_occup_grid_sseg = sg_map_global.rotate_map(grid=step_occup_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                step_segm_grid_sseg = sg_map_global.rotate_map(grid=step_segm_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                # Crop the grid around the agent at each timestep
                step_occup_grid_maps = utils.crop_grid(grid=step_occup_grid_sseg, crop_size=map_config['grid_dim'])
                step_segm_grid_maps = utils.crop_grid(grid=step_segm_grid_sseg, crop_size=map_config['grid_dim'])

                batch_gaussian_heatmaps = []
                for b in range(self.batch_size):
                    gaussian_heatmaps = utils.locs_to_heatmaps(torch.tensor(batch_candidate_rel_pos[b]).to(self.device), map_config['grid_dim'], (map_config['heatmap_size'],map_config['heatmap_size']), sigma=5.)
                    gaussian_heatmaps = gaussian_heatmaps.sum(0,keepdim=True) / gaussian_heatmaps.shape[0]
                    batch_gaussian_heatmaps.append(gaussian_heatmaps)

                batch_gaussian_heatmaps = torch.cat(batch_gaussian_heatmaps,dim=0)

                if self.config.GPU_NUMBERS > 1:
                    predicted_occup_grid_maps =  self.policy.net.module.occupancy_map_predictor(step_occup_grid_maps.unsqueeze(1))
                    step_segm_occup_grid_maps = torch.cat((step_segm_grid_maps,predicted_occup_grid_maps),dim=-3)
                    predicted_segm_grid_maps = self.policy.net.module.semantic_map_predictor(step_segm_occup_grid_maps.unsqueeze(1))
                    step_segm_occup_grid_maps = torch.cat((predicted_segm_grid_maps.unsqueeze(1),predicted_occup_grid_maps.unsqueeze(1)),dim=-3)
                    waypoint_grid_maps = self.policy.net.module.waypoint_predictor(step_segm_occup_grid_maps).squeeze(1)
                else:
                    predicted_occup_grid_maps =  self.policy.net.occupancy_map_predictor(step_occup_grid_maps.unsqueeze(1))
                    step_segm_occup_grid_maps = torch.cat((step_segm_grid_maps,predicted_occup_grid_maps),dim=-3)
                    predicted_segm_grid_maps = self.policy.net.semantic_map_predictor(step_segm_occup_grid_maps.unsqueeze(1))
                    step_segm_occup_grid_maps = torch.cat((predicted_segm_grid_maps.unsqueeze(1),predicted_occup_grid_maps.unsqueeze(1)),dim=-3)
                    waypoint_grid_maps = self.policy.net.waypoint_predictor(step_segm_occup_grid_maps).squeeze(1)

                loss += self.cross_entropy_loss(input=predicted_occup_grid_maps,target=batch_gt_map_occupancy.to(torch.int64))
                loss += self.cross_entropy_loss(input=predicted_segm_grid_maps,target=batch_gt_map_semantic.to(torch.int64))
                loss += self.mse_loss(waypoint_grid_maps,batch_gaussian_heatmaps) * 10. # scale

                for b in range(self.batch_size):
                    
                    rel_heading, rel_elevation, _ = calculate_vp_rel_pos_fts(positions[b],next_pos[b],headings[b],0.)

                    ###############
                    rel_heading += random.uniform(-math.pi,math.pi) # Data Augmentation
                    ###############

                    next_heading = rel_heading + headings[b] + math.pi

                    q1 = math.cos(next_heading/2)
                    q2 = math.sin(next_heading/2)

                    rotation = np.quaternion(q1,0,q2,0)

                    camera_obs = self.envs.call_at(b, "get_observation",{"source_position":next_pos[b],"source_rotation":rotation,"keep_agent_at_new_pose":True})

                    observations[b] = camera_obs
                           
            if mode == 'eval':
                with torch.no_grad():

                    # agent's current position and heading
                    for ob_i in range(self.batch_size):
                        agent_state_i = self.envs.call_at(ob_i,
                                "get_agent_info", {})
                        positions[b] = agent_state_i['position']
                        headings[b] = agent_state_i['heading']

                    batch_img = []
                    batch_depth = []
                    batch_local3D_step = []
                    batch_rel_abs_pose = []
                    batch_candidate_rel_pos = []
                    batch_gt_map_semantic = []
                    batch_gt_map_occupancy = []

                    for b in range(self.batch_size):

                        _, candidate_pos = get_gt_nav_map(self.navigability_map,scene_ids,waypoint_ids)
                        #####################################


                        img = observations[b]['rgb']
                        depth = observations[b]['depth'].reshape(map_config['img_size'][0], map_config['img_size'][1], 1)

                        viz_img = img
                        img = torch.tensor(img).to(self.device)
                        depth = torch.tensor(depth).to(self.device)

                        if map_config['norm_depth']:
                            depth_abs = utils.unnormalize_depth(depth, min=0.0, max=10.0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                        batch_img.append(img.unsqueeze(0))
                        batch_depth.append(depth_abs.unsqueeze(0))

                        local3D_step = utils.depth_to_3D(depth_abs, map_config['img_size'], xs, ys, inv_K)
                        batch_local3D_step.append(local3D_step)

                        agent_state = self.envs.call_at(b,"get_agent_state", {})

                        agent_pose, y_height = utils.get_sim_location(agent_state=agent_state)
                        abs_poses[b].append(agent_pose)

                        scene_id = scene_ids[b]
                        # get gt map from agent pose for visualization later (pose is at the center looking upwards)
                        x, y, z, label_seq, color_pcloud = utils.slice_scene(x=self.pclouds[scene_id][0].copy(),
                                                                            y=self.pclouds[scene_id][1].copy(),
                                                                            z=self.pclouds[scene_id][2].copy(),
                                                                            label_seq=self.label_seq_objects[scene_id].copy(),
                                                                            height=y_height,
                                                                            color_pcloud=self.color_pcloud[scene_id])

                        gt_map_semantic, gt_map_occupancy, _ = utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[b][stepk],
                                                                    grid_dim=map_config['grid_dim'], cell_size=map_config['cell_size'], color_pcloud=color_pcloud, z=z) 
                        batch_gt_map_semantic.append(gt_map_semantic)
                        batch_gt_map_occupancy.append(gt_map_occupancy)

                        try:
                            next_pos[b] = random.choice(candidate_pos[b])
                            waypoint_ids[b] = self.position_to_waypoint_id[scene_ids[b]][str(next_pos[b])]
                            visited_waypoints[b].append(waypoint_ids[b])
                            next_pos[b] = self.waypoint_id_to_position[scene_ids[b]]['nodes'][waypoint_ids[b]]
                            if waypoint_ids[b] in visited_waypoints[b]:
                                next_pos[b] = random.choice(candidate_pos[b])
                                waypoint_ids[b] = self.position_to_waypoint_id[scene_ids[b]][str(next_pos[b])]
                                visited_waypoints[b].append(waypoint_ids[b])
                                next_pos[b] = self.waypoint_id_to_position[scene_ids[b]]['nodes'][waypoint_ids[b]]
                        
                        except:
                            print('Position error! Skip...')
                            for i in range(len(candidate_pos[b])):
                                if str(candidate_pos[b][i]) in self.position_to_waypoint_id[scene_ids[b]]:
                                    next_pos[b] = candidate_pos[b][i]
                                    waypoint_ids[b] = self.position_to_waypoint_id[scene_ids[b]][str(next_pos[b])]
                                    visited_waypoints[b].append(waypoint_ids[b])
                                    next_pos[b] = self.waypoint_id_to_position[scene_ids[b]]['nodes'][waypoint_ids[b]]
                                    break

                        candidate_rel_pos = []
                        for candidate_pos_item in candidate_pos[b]:
                            candidate_rel_pos_item, candidate_visible = utils.transform_to_map_coords(sg=sg_map, position=candidate_pos_item, abs_pose=abs_poses[b][stepk],grid_size=map_config['grid_dim'][0],cell_size=map_config['cell_size'],device=self.device)
                            candidate_rel_pos.append(candidate_rel_pos_item[0][0].numpy().tolist())

                        batch_candidate_rel_pos.append(candidate_rel_pos)
                        # Keep track of the agent's relative pose from the initial position
                        rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[b][stepk], pos1=abs_poses[b][0])
                        _rel_abs_pose = torch.Tensor(rel_abs_pose).unsqueeze(0).float()
                        _rel_abs_pose = _rel_abs_pose.to(self.device)
                        batch_rel_abs_pose.append(_rel_abs_pose)


                    ### Run the img segmentation model to get the ground-projected semantic segmentation
                    batch_abs_poses = torch.tensor(abs_poses).to(self.device)
                    batch_rel_abs_pose = torch.cat(batch_rel_abs_pose,dim=0)
                    batch_gt_map_semantic = torch.cat(batch_gt_map_semantic,dim=0)
                    batch_gt_map_occupancy = torch.cat(batch_gt_map_occupancy,dim=0)

                    batch_img = torch.cat(batch_img,dim=0)
                    batch_depth = torch.cat(batch_depth,dim=0)
                    depth_img = batch_depth.clone().permute(0,3,1,2)

                    depth_img = F.interpolate(depth_img, size=map_config['img_segm_size'], mode='nearest')
                    imgData = utils.preprocess_img(batch_img, cropSize=map_config['img_segm_size'], pixFormat='NCHW', normalize=True)

                    segm_batch = {'images':imgData.to(self.device).unsqueeze(1),
                                'depth_imgs':depth_img.to(self.device).unsqueeze(1)}

                    pred_ego_sseg, img_segm = utils.run_img_segm(model=self.img_segmentor, 
                                                            input_batch=segm_batch, 
                                                            object_labels=map_config['object_labels'], 
                                                            crop_size=map_config['global_dim'], 
                                                            cell_size=map_config['cell_size'],
                                                            xs=self._xs,
                                                            ys=self._ys,
                                                            inv_K=inv_K,
                                                            points2D_step=self._points2D_step)   


                    # do ground-projection, update the projected map
                    ego_grid_sseg_3 = utils.est_occ_from_depth(batch_local3D_step, grid_dim=map_config['global_dim'], cell_size=map_config['cell_size'], 
                                                                                    device=self.device, occupancy_height_thresh=map_config['occupancy_height_thresh'])

                    # Transform the ground projected egocentric grids to geocentric using relative pose
                    occup_grid_sseg = sg_map_global.spatialTransformer(grid=ego_grid_sseg_3, pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                    semantic_grid_sseg = sg_map_global.spatialTransformer(grid=pred_ego_sseg[:,0], pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                    # step_geo_grid contains the map snapshot every time a new observation is added
                    step_occup_grid_sseg, step_segm_grid_sseg = sg_map_global.update_proj_grid_bayes(occup_grid_sseg.unsqueeze(1),semantic_grid_sseg.unsqueeze(1))
                    # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)

                    step_occup_grid_sseg = sg_map_global.rotate_map(grid=step_occup_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                    step_segm_grid_sseg = sg_map_global.rotate_map(grid=step_segm_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                    # Crop the grid around the agent at each timestep
                    step_occup_grid_maps = utils.crop_grid(grid=step_occup_grid_sseg, crop_size=map_config['grid_dim'])
                    step_segm_grid_maps = utils.crop_grid(grid=step_segm_grid_sseg, crop_size=map_config['grid_dim'])

                    batch_gaussian_heatmaps = []
                    for b in range(self.batch_size):
                        gaussian_heatmaps = utils.locs_to_heatmaps(torch.tensor(batch_candidate_rel_pos[b]).to(self.device), map_config['grid_dim'], (map_config['heatmap_size'],map_config['heatmap_size']), sigma=5.)
                        gaussian_heatmaps = gaussian_heatmaps.sum(0,keepdim=True) / gaussian_heatmaps.shape[0]
                        batch_gaussian_heatmaps.append(gaussian_heatmaps)

                    batch_gaussian_heatmaps = torch.cat(batch_gaussian_heatmaps,dim=0)                  

                    if self.config.GPU_NUMBERS > 1:
                        predicted_occup_grid_maps =  self.policy.net.module.occupancy_map_predictor(step_occup_grid_maps.unsqueeze(1))
                        step_segm_occup_grid_maps = torch.cat((step_segm_grid_maps,predicted_occup_grid_maps),dim=-3)
                        predicted_segm_grid_maps = self.policy.net.module.semantic_map_predictor(step_segm_occup_grid_maps.unsqueeze(1))
                        step_segm_occup_grid_maps = torch.cat((predicted_segm_grid_maps.unsqueeze(1),predicted_occup_grid_maps.unsqueeze(1)),dim=-3)
                        waypoint_grid_maps = self.policy.net.module.waypoint_predictor(step_segm_occup_grid_maps).squeeze(1)
                    else:
                        predicted_occup_grid_maps =  self.policy.net.occupancy_map_predictor(step_occup_grid_maps.unsqueeze(1))
                        step_segm_occup_grid_maps = torch.cat((step_segm_grid_maps,predicted_occup_grid_maps),dim=-3)
                        predicted_segm_grid_maps = self.policy.net.semantic_map_predictor(step_segm_occup_grid_maps.unsqueeze(1))
                        step_segm_occup_grid_maps = torch.cat((predicted_segm_grid_maps.unsqueeze(1),predicted_occup_grid_maps.unsqueeze(1)),dim=-3)
                        waypoint_grid_maps = self.policy.net.waypoint_predictor(step_segm_occup_grid_maps).squeeze(1)

                    for b in range(self.batch_size):
                        waypoint_grid_maps[b] = waypoint_grid_maps[b] - waypoint_grid_maps[b].min()

                    #waypoint_grid_maps = self.noise_filter(waypoint_grid_maps).squeeze(1)

                    x = torch.arange(0, map_config['grid_dim'][0], dtype=torch.float32).to(self.device)
                    y = torch.arange(0, map_config['grid_dim'][1], dtype=torch.float32).to(self.device)
                    yg, xg = torch.meshgrid(y,x)
                    yg = -(yg.to(self.device) -  map_config['grid_dim'][1] / 2. + 0.5)
                    xg = xg.to(self.device) -  map_config['grid_dim'][0] / 2. + 0.5

                    grid_rel_angle = torch.atan2(xg, yg)
                    grid_rel_angle = (grid_rel_angle + 2*math.pi) % (2.*math.pi)

                    predicted_waypoints = [[] for direction_idx in range(12)]
                    for direction_idx in range(12):
                        back_angle = math.radians(direction_idx * 30.- 15.) 
                        front_angle = math.radians(direction_idx * 30.+ 15.)
                        if direction_idx == 0:
                            back_angle += 2.*math.pi
                            selected_part = (back_angle <= grid_rel_angle) | (grid_rel_angle <= front_angle)
                        else:
                            selected_part = (back_angle <= grid_rel_angle) & (grid_rel_angle <= front_angle)

                        tmp_waypoint_grid_maps = waypoint_grid_maps.clone()
                        tmp_waypoint_grid_maps[:,selected_part==False] = 0.
                        max_value, max_index = tmp_waypoint_grid_maps.view(self.batch_size,-1).max(dim=-1)
                        max_y = torch.div(max_index, map_config['grid_dim'][0], rounding_mode='floor')
                        max_x = max_index % map_config['grid_dim'][0]

                        predicted_waypoints[direction_idx] = torch.cat([max_value.view(self.batch_size,1),max_x.view(self.batch_size,1),max_y.view(self.batch_size,1)],dim=-1).unsqueeze(1)

                    predicted_waypoints = torch.cat(predicted_waypoints,dim=1)

                    # merge near waypoints
                    merge_scale = 8
                    for x_merge in range(2):
                        for y_merge in range(2):
                            tmp_predicted_waypoints = predicted_waypoints[:,:,1:].to(torch.int64)  
                            if x_merge == 1:
                                tmp_predicted_waypoints[:,:,0] = tmp_predicted_waypoints[:,:,0] + merge_scale
                            if y_merge == 1:
                                tmp_predicted_waypoints[:,:,1] = tmp_predicted_waypoints[:,:,1] + merge_scale

                            tmp_predicted_waypoints = torch.div(tmp_predicted_waypoints, merge_scale*2, rounding_mode='floor').to(torch.int32)
                            for b in range(batch_size):
                                tmp_dict = {}
                                for i in range(12):
                                    # delete occupied waypoints
                                    if predicted_occup_grid_maps[b,:,predicted_waypoints[b,i,1].to(torch.int64),predicted_waypoints[b,i,2].to(torch.int64)].argmax().cpu().item() == 1: # occupied
                                        predicted_waypoints[b,i,0] = 0.

                                    key = str([tmp_predicted_waypoints[b][i][0].cpu().item(), tmp_predicted_waypoints[b][i][1].cpu().item()])
                                    if key in tmp_dict:
                                        if predicted_waypoints[b,tmp_dict[key],0] > predicted_waypoints[b,i,0]:
                                            predicted_waypoints[b,i,0] = 0.
                                        else:
                                            predicted_waypoints[b,tmp_dict[key],0] = 0.
                                    else:
                                        tmp_dict[key] = i

    
                    # select k waypoints
                    selected_waypoint_index = torch.topk(predicted_waypoints[:,:,0], k=8, dim=-1, largest=True)[1]
                    selected_waypoints = [0 for b in range(batch_size)]
                    batch_angle_idxes = []
                    batch_distance_idxes = []
                    for b in range(batch_size):
                        selected_waypoints[b] = predicted_waypoints[b,selected_waypoint_index[b]]
                        selected_waypoints[b] = selected_waypoints[b][selected_waypoints[b][:,0]!=0]
                        selected_waypoints[b] = selected_waypoints[b][:,1:]
                        rel_y = -(selected_waypoints[b][:,1] - map_config['grid_dim'][1]//2 + 0.5) * 0.05
                        rel_x = (selected_waypoints[b][:,0] - map_config['grid_dim'][0]//2 + 0.5) * 0.05
                        rel_angle = torch.atan2(rel_x, rel_y)

                        rel_dist = torch.sqrt(torch.square(rel_x) + torch.square(rel_y))
                        mask = (0.1 < rel_dist) & (rel_dist < 5.)
                        rel_dist = rel_dist[mask]
                        rel_angle = (rel_angle + 2*math.pi) % (2.*math.pi)
                        rel_angle = rel_angle[mask]
                        selected_waypoints[b] = selected_waypoints[b][mask]

                        # Discretization
                        angle_idx = torch.div((rel_angle+(math.pi/120)), (math.pi/60), rounding_mode='floor').to(torch.int32)
                        distance_idx = torch.div(rel_dist+0.25/2., 0.25, rounding_mode='floor').to(torch.int32) 

                        batch_angle_idxes.append(angle_idx)
                        batch_distance_idxes.append(distance_idx)


                    #import matplotlib.pyplot as plt
                    for b in range(self.batch_size):                
                        #plt.cla()
                        #plt.imshow(waypoint_grid_maps[b].cpu().numpy())

                        #fig = plt.gcf()
                        #plt.margins(0,0)
                        #fig.savefig(str(b)+'_'+str(stepk)+'reponse_map.png', dpi=500, bbox_inches='tight')
                        #plt.show()

                        ################################## 
                        #save_img_dir = "example/" + scene_id + '/'
                        #viz_utils.write_tensor_imgSegm(step_segm_grid_maps, save_img_dir, name="img_segm", t=stepk, labels=27)

                        #save_img_dir = "example/" + scene_id + '/occup_' + str(b) +'_'
                        #viz_utils.write_tensor_imgSegm(predicted_occup_grid_maps, save_img_dir, name="img_segm", t=stepk, labels=3)


                        #save_img_dir = "example/" + scene_id + '/img_' + str(b) +'_'
                        #im_path = save_img_dir + "_" + str(stepk) + ".png"

                        #Image.fromarray(viz_img).save(im_path)
                                       

                        #save_img_dir = "example/" + scene_id + '/predicted_cwp_' + str(b) +'_'
                        #viz_utils.write_tensor_imgSegm(predicted_segm_grid_maps, save_img_dir, name="img_segm", t=stepk, labels=27, waypoints=selected_waypoints[b])
                        ##################################
                    

                        rel_heading, rel_elevation, _ = calculate_vp_rel_pos_fts(positions[b],next_pos[b],headings[b],0.)

                        next_heading = rel_heading + headings[b] + math.pi

                        q1 = math.cos(next_heading/2)
                        q2 = math.sin(next_heading/2)

                        rotation = np.quaternion(q1,0,q2,0)

                        camera_obs = self.envs.call_at(b, "get_observation",{"source_position":next_pos[b],"source_rotation":rotation,"keep_agent_at_new_pose":True})

                        observations[b] = camera_obs

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        #for b in range(self.batch_size):
         #       sim = sims[b]
          #      sim.close()

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            try:
                self.logs['IL_loss'].append(loss.item())
            except:
                self.logs['IL_loss'].append(loss)


