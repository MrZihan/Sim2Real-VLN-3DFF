import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy
import tqdm
from PIL import Image
from vlnce_baselines.common.aux_losses import AuxLosses

from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST, calculate_vp_rel_pos_fts
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from fastdtw import fastdtw


import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
import cv2

import vlnce_baselines.waypoint_networks.utils as utils
from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele )
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from vlnce_baselines.waypoint_networks.semantic_grid import SemanticGrid
from vlnce_baselines.waypoint_networks import get_img_segmentor_from_options
from vlnce_baselines.waypoint_networks.resnetUnet import ResNetUNet
import vlnce_baselines.waypoint_networks.viz_utils as viz_utils
from vlnce_baselines.models.Policy_ViewSelection_ETP import Net
import socket
import pickle
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import clip


class TCPClient:
    def __init__(self,host:str,port:int):
        self.client=socket.socket()
        start_state = False
        while start_state==False:
            try:
                self.client.connect((host,port))
                start_state = True
                print('Client Start.')
            except:
                time.sleep(1)

    def send_data(self,data:bytes):
        data = pickle.dumps(data)
        self.client.send(pickle.dumps(len(data)).ljust(64))
        self.client.sendall(data)

    def close(self):
        self.client.close()

class TCPServer:
    def __init__(self,host:str,port:int):
        self.server=socket.socket()
        start_state = False
        while start_state==False:
            try:
                self.server.bind((host,port))
                self.server.listen(1)
                start_state = True
                print('Server Start.')
            except:
                time.sleep(1)
        self.client_socket, self.clientAddr = self.server.accept()

    def recv_data(self):
        try:
            data_len = self.client_socket.recv(64)
        except:
            time.sleep(1)
        data_len = pickle.loads(data_len)
        buffer = b"" 
        while True:
            received_data = self.client_socket.recv(512)
            buffer = buffer + received_data 
            if len(buffer) == data_len: 
                break
        data = pickle.loads(buffer) 
        return data

    def close(self):
        self.server.close()




class RLTrainer():
    def __init__(self, config=None):
        self.device = 'cuda'
        Server_IP = config['Server_IP']
        Robot_IP = config['Robot_IP']
        self.rgb_server=TCPServer(Server_IP,5001)
        self.depth_server=TCPServer(Server_IP,5002)
        self.location_server=TCPServer(Server_IP,5003)
        self.action_client=TCPClient(Robot_IP,5000)
        self.max_len = int(config['IL']['max_traj_len']) #  * 0.97 transfered gt path got 0.96 spl
        self.config = config
        self.batch_size = 1


    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _initialize_policy(
        self, config,
        load_from_ckpt: bool
    ):
        self.policy = Net(config['MODEL'])
        ''' initialize the waypoint predictor here '''

        self.tokenizer = AutoTokenizer.from_pretrained('bert_config/bert-base-uncased')


        n_object_classes = 27
        ## Load the pre-trained img segmentation model
        self.img_segmentor = get_img_segmentor_from_options(n_object_classes,1.0)
        self.img_segmentor = self.img_segmentor.to(self.device)
     
        self.img_segmentor = torch.nn.DataParallel(self.img_segmentor)

        checkpoint = torch.load("pretrained/segm.pt")
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

        self.policy.to(self.device)
        self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                device_ids=[self.device], output_device=self.device)

        

        ckpt_dict = self.load_checkpoint('pretrained/cwp_predictor.pth', map_location="cpu")           
        b = [key for key in ckpt_dict["state_dict"].keys()]
        for key in b:
            if 'rgb_encoder' in key:
                ckpt_dict['state_dict'].pop(key) 
        self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
      

        ckpt_dict = self.load_checkpoint('pretrained/NeRF_p16_8x8.pth', map_location="cpu")   
        b = [key for key in ckpt_dict["state_dict"].keys()]
        for key in b:
            if 'rgb_encoder' in key:
                ckpt_dict['state_dict'].pop(key) 
        self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)

        if load_from_ckpt:
            ckpt_dict = self.load_checkpoint(config['IL']['ckpt_to_load'], map_location="cpu")           
            self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
            start_iter = ckpt_dict["iteration"]          

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
       
        return start_iter



    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_loc_fts = [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.batch_size):
            rgb_fts, loc_fts , nav_types = [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool_)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'loc_fts': batch_loc_fts,
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

        bs = len(cur_vp)
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

        

    @torch.no_grad()
    def rollout(self, mode, instruction):

        self.policy.net.module.rgb_encoder.eval()
        self.policy.net.module.occupancy_map_predictor.eval()
        self.policy.net.module.semantic_map_predictor.eval()
        self.policy.net.module.waypoint_predictor.eval()

        # encode instructions
        all_txt_ids = self.tokenizer(instruction)['input_ids']
        all_txt_ids = torch.tensor([all_txt_ids])
        instr_pad_id = 1 if self.config['MODEL']['task_type'] == 'rxr' else 0
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        total_actions = 0.
        not_done_index = list(range(self.batch_size))
        have_real_pos = False
        ghost_aug = 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config['IL']['loc_noise'], 
                               self.config['MODEL']['merge_ghost'],
                               ghost_aug) for _ in range(self.batch_size)]
        prev_vp = [None] * self.batch_size


        ##############
        total_actions = 0.

        hfov = 69. * np.pi / 180.
        vfov = 42. * np.pi / 180.
        map_config={'hfov':hfov,'vfov':vfov,'global_dim':(512,512),'grid_dim':(192,192),'heatmap_size':192,'cell_size':0.05,'img_segm_size':(128,128),'spatial_labels':3,'object_labels':27,'img_raw_size':(480,640),'img_size':[224,224],  'occupancy_height_thresh':0.4,'norm_depth':True}
        # 3d info
        xs, ys = torch.tensor(np.array(np.meshgrid(np.linspace(-1,1,map_config['img_segm_size'][0]), np.linspace(1,-1,map_config['img_segm_size'][1]))), device='cuda')

        xs = xs.reshape(1,map_config['img_segm_size'][0],map_config['img_segm_size'][1])
        ys = ys.reshape(1,map_config['img_segm_size'][0],map_config['img_segm_size'][1])
        K = np.array([
            [1 / np.tan(map_config['hfov'] / 2.), 0., 0., 0.],
            [0., 1 / np.tan(map_config['vfov'] / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        inv_K = torch.tensor(np.linalg.inv(K), device=self.device)


        # For each episode we need a new instance of a fresh global grid
        sg_map_global = SemanticGrid(self.batch_size, map_config['global_dim'], map_config['heatmap_size'], map_config['cell_size'],
                            spatial_labels=map_config['spatial_labels'], object_labels=map_config['object_labels'])


        abs_poses = [[] for b in range(self.batch_size)]
        positions = [None for b in range(self.batch_size)]
        headings = [None for b in range(self.batch_size)]
        observations = [{} for b in range(self.batch_size)]
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module
 
        batch_size = self.batch_size
        wait_for_move = None
        observe_environment = 0
        collision_distance = 5.
        for stepk in range(self.max_len): #self.max_len
            # agent's current position and heading   

            observe_environment += 1
            for ob_i in range(batch_size):
                img = torch.tensor(self.rgb_server.recv_data())              
                depth = torch.tensor(self.depth_server.recv_data())
                location = torch.tensor(self.location_server.recv_data())
                positions[ob_i] = np.concatenate([location[0:1],torch.tensor([0.88]),location[1:2]],0) # Robot height 0.88m
                headings[ob_i] = (2*math.pi+location[2].item())%(2*math.pi)

                observations[ob_i]['rgb'] = F.interpolate(img.unsqueeze(0).permute(0,3,1,2), size=map_config['img_size'], mode='nearest').permute(0,2,3,1).squeeze(0)
                observations[ob_i]['depth'] =  F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=map_config['img_segm_size'], mode='nearest').squeeze(0).squeeze(0)
                collision_distance = depth[depth>0.01].min().item()

            if stepk == 0:
                policy_net.start_positions = positions
                policy_net.start_headings = [(heading+2*math.pi)%(2*math.pi) for heading in headings]
                policy_net.global_fts = [[] for i in range(batch_size)]
                policy_net.global_position_x = [[] for i in range(batch_size)]
                policy_net.global_position_y = [[] for i in range(batch_size)]
                policy_net.global_position_z = [[] for i in range(batch_size)]
                policy_net.global_patch_scales = [[] for i in range(batch_size)]
                policy_net.global_patch_directions = [[] for i in range(batch_size)]
                policy_net.global_mask = [[] for i in range(batch_size)]

            policy_net.action_step = stepk + 1
            policy_net.positions = positions
            policy_net.headings = [(heading+2*math.pi)%(2*math.pi) for heading in headings]

            with torch.no_grad():              
                batch_img = []
                batch_depth = []
                batch_local3D_step = []
                batch_rel_abs_pose = []

                for b in range(batch_size):
                    
                    ##################################
                    img = observations[b]['rgb'].to(self.device)
                    depth = observations[b]['depth'].to(self.device).to(torch.float32)
                    q1 = math.cos(headings[b]/2)
                    q2 = math.sin(headings[b]/2)
                    rotation = np.quaternion(q1,0,q2,0)
                    agent_state = {
                        'position': positions[b],
                        'rotation': rotation
                        }

                    ################
                    policy_net.positions[b] = positions[b] #!!!!!!!!!!!!!!!!!!!!
                    policy_net.headings[b] = headings[b]   #!!!!!!!!!!!!!!!!!!!!
                    ################
                    viz_img = img.cpu().numpy()
                    viz_depth = depth.cpu().numpy()
                    depth_abs = depth.unsqueeze(-1)
                    #if map_config['norm_depth']:
                    #    depth_abs = utils.unnormalize_depth(depth, min=0.0, max=10.0) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    batch_img.append(img.unsqueeze(0))
                    batch_depth.append(depth_abs.unsqueeze(0))

                    local3D_step = utils.depth_to_3D(depth_abs, map_config['img_segm_size'], xs, ys, inv_K)
                    batch_local3D_step.append(local3D_step)

                    agent_pose, y_height = utils.get_sim_location(agent_state=agent_state)
                        
                    if len(abs_poses[b]) < stepk+1:
                        abs_poses[b].append(agent_pose)
                    else:
                        abs_poses[b][stepk] = agent_pose


                    # Keep track of the agent's relative pose from the initial position
                    rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[b][stepk], pos1=abs_poses[b][0])
                    _rel_abs_pose = torch.Tensor(rel_abs_pose).unsqueeze(0).float()
                    _rel_abs_pose = _rel_abs_pose.to(self.device)
                    batch_rel_abs_pose.append(_rel_abs_pose)

                if batch_rel_abs_pose != []:
                    ### Run the img segmentation model to get the ground-projected semantic segmentation
                    batch_abs_poses = torch.tensor(abs_poses).to(self.device)
                    batch_rel_abs_pose = torch.cat(batch_rel_abs_pose,dim=0)

                    batch_img = torch.cat(batch_img,dim=0)
                        
                    batch_depth = torch.cat(batch_depth,dim=0)
                    depth_img = batch_depth.clone().permute(0,3,1,2)

                    #depth_img = F.interpolate(depth_img, size=map_config['img_segm_size'], mode='nearest')
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

            
                ## update the feature field
                clip_fts = self.policy.net(
                    mode = "feature_field",
                    observations = observations,
                    in_train = False,
                )

            

                #########################################################

                # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
                step_occup_grid_sseg = sg_map_global.rotate_map(grid=step_occup_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)
                step_segm_grid_sseg = sg_map_global.rotate_map(grid=step_segm_grid_sseg.squeeze(1), rel_pose=batch_rel_abs_pose, abs_pose=batch_abs_poses)

                # Crop the grid around the agent at each timestep
                step_occup_grid_maps = utils.crop_grid(grid=step_occup_grid_sseg, crop_size=map_config['grid_dim'])
                step_segm_grid_maps = utils.crop_grid(grid=step_segm_grid_sseg, crop_size=map_config['grid_dim'])               

                predicted_occup_grid_maps =  self.policy.net.module.occupancy_map_predictor(step_occup_grid_maps.unsqueeze(1))
                step_segm_occup_grid_maps = torch.cat((step_segm_grid_maps,predicted_occup_grid_maps),dim=-3)
                predicted_segm_grid_maps = self.policy.net.module.semantic_map_predictor(step_segm_occup_grid_maps.unsqueeze(1))
                step_segm_occup_grid_maps = torch.cat((predicted_segm_grid_maps.unsqueeze(1),predicted_occup_grid_maps.unsqueeze(1)),dim=-3)
                waypoint_grid_maps = self.policy.net.module.waypoint_predictor(step_segm_occup_grid_maps).view(batch_size,1,map_config['grid_dim'][0],map_config['grid_dim'][1]).squeeze(1)

                
                for b in range(self.batch_size):     
                    plt.ion()
                    plt.clf()
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
                    plt.subplot(2, 2, 1)
                    plt.imshow(viz_img)
                    fig = plt.gcf()
                    plt.title('RGB Observation')

                    ##################################

                    plt.subplot(2, 2, 2)
                    viz_depth = cv2.applyColorMap((viz_depth/5.* 255).astype(np.uint8), cv2.COLORMAP_JET)
                    plt.imshow(viz_depth)
                    fig = plt.gcf()
                    plt.title('Depth Observation')


                    occup_image = viz_utils.get_tensor_imgSegm(predicted_occup_grid_maps, labels=3)
                    plt.subplot(2, 2, 3)
                    plt.imshow(occup_image)
                    fig = plt.gcf()
                    plt.title('Occupancy map')


                    semantic_image = viz_utils.get_tensor_imgSegm(predicted_segm_grid_maps, labels=27)
                    plt.subplot(2, 2, 4)
                    plt.imshow(semantic_image)
                    fig = plt.gcf()
                    plt.title('Semantic map')
                    plt.pause(0.001)



                cls_fts = clip_fts[0,0:1]
                region_fts = clip_fts[0,1:]

                clip_distance = F.interpolate(depth_img, size=(14,14), mode='nearest').view(14*14)

                text = clip.tokenize(' '.join(instruction.split(' ')[-7:])).to("cuda")
                text_fts = policy_net.rgb_encoder.model.encode_text(text).detach()
                cls_fts = cls_fts / torch.linalg.norm(cls_fts, dim=-1, keepdim=True)
                region_fts = region_fts / torch.linalg.norm(region_fts, dim=-1, keepdim=True)
                text_fts = text_fts / torch.linalg.norm(text_fts, dim=-1, keepdim=True)

                cls_scores = (cls_fts @ text_fts.T).squeeze(-1)
                region_scores = (region_fts @ text_fts.T).squeeze(-1)
                    

                if ((0.1 < clip_distance[region_scores > 0.3]) & (clip_distance[region_scores > 0.3] < 0.5)).any() and ((cls_scores > 0.29).any() and collision_distance < 1.):
                    print('Stop.')
                    exit()

                if (region_scores > 0.3).any() or (cls_scores > 0.29).any():
                    observe_environment -= 1
                    self.single_move_control(collision_distance/2.)
                    continue


                if observe_environment < 9:
                    self.turn_for_observe()
                    continue

                if wait_for_move != None:
                    wait_for_move = self.single_move_control(min(wait_for_move,collision_distance/2.))
                    continue


                for b in range(batch_size):
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
                    max_value, max_index = tmp_waypoint_grid_maps.view(batch_size,-1).max(dim=-1)
                    max_y = torch.div(max_index, map_config['grid_dim'][0], rounding_mode='floor')
                    max_x = max_index % map_config['grid_dim'][0]

                    predicted_waypoints[direction_idx] = torch.cat([max_value.view(batch_size,1),max_x.view(batch_size,1),max_y.view(batch_size,1)],dim=-1).unsqueeze(1)

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
                                #if predicted_occup_grid_maps[b,:,predicted_waypoints[b,i,1].to(torch.int64),predicted_waypoints[b,i,2].to(torch.int64)].argmax().cpu().item() == 1: # occupied
                                #    predicted_waypoints[b,i,0] = 0.

                                key = str([tmp_predicted_waypoints[b][i][0].cpu().item(), tmp_predicted_waypoints[b][i][1].cpu().item()])
                                if key in tmp_dict:
                                    if predicted_waypoints[b,tmp_dict[key],0] > predicted_waypoints[b,i,0]:
                                        predicted_waypoints[b,i,0] = 0.
                                    else:
                                        predicted_waypoints[b,tmp_dict[key],0] = 0.
                                else:
                                    tmp_dict[key] = i


    
                # select k waypoints
                selected_waypoint_index = torch.topk(predicted_waypoints[:,:,0], k=12, dim=-1, largest=True)[1]
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

                    rel_angle = (rel_angle + 2*math.pi) % (2.*math.pi)
                    rel_dist = torch.sqrt(torch.square(rel_x) + torch.square(rel_y))

                    # Discretization
                    angle_idx = torch.div((rel_angle+(math.pi/120)), (math.pi/60), rounding_mode='floor').to(torch.int32)
                    distance_idx = torch.div(rel_dist+0.25/2., 0.25, rounding_mode='floor').to(torch.int32)

                    batch_angle_idxes.append(angle_idx)
                    batch_distance_idxes.append(distance_idx)


                ###############################
                total_actions += self.batch_size
                txt_masks = all_txt_masks[not_done_index]
                txt_embeds = all_txt_embeds[not_done_index]

                # cand waypoint representation, need to be freezed
                wp_outputs = self.policy.net(
                    mode = "waypoint",
                    batch_angle_idxes = batch_angle_idxes,
                    batch_distance_idxes = batch_distance_idxes,
                    observations = observations,
                    in_train = False,
                )

            if observe_environment >= 9:
                for b in range(self.batch_size):     
                    plt.ion()
                    plt.clf()
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
                    plt.subplot(2, 2, 1)
                    plt.imshow(viz_img)
                    fig = plt.gcf()
                    plt.title('RGB Observation')

                    ##################################

                    plt.subplot(2, 2, 2)
                    viz_depth = cv2.applyColorMap((viz_depth/5.* 255).astype(np.uint8), cv2.COLORMAP_JET)
                    plt.imshow(viz_depth)
                    fig = plt.gcf()
                    plt.title('Depth Observation')


                    occup_image = viz_utils.get_tensor_imgSegm(predicted_occup_grid_maps, labels=3)
                    plt.subplot(2, 2, 3)
                    plt.imshow(occup_image)
                    fig = plt.gcf()
                    plt.title('Occupancy map')


                    semantic_image = viz_utils.get_tensor_imgSegm(predicted_segm_grid_maps, labels=27, waypoints=selected_waypoints[b].cpu().to(torch.int64).numpy().tolist())
                    plt.subplot(2, 2, 4)
                    plt.imshow(semantic_image)
                    fig = plt.gcf()
                    plt.title('Semantic map')
                    plt.pause(0.001)

                    ##################################


            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_node
            cur_pos, cur_ori = positions, headings
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.batch_size):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)


            for i in range(self.batch_size):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                            cur_vp[i], cur_pos[i], cur_embeds,
                                            cand_vp[i], cand_pos[i], cand_embeds)

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            #nav_logits[:,0] += 0.5
            a_t = nav_logits.argmax(dim=-1)
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config['IL']['back_algo'] == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    print("Stop.")
                    exit()
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    
                    vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config['IL']['back_algo'] == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config['MODEL']['consume_ghost']:
                        gmap.delete_ghost(ghost_vp)

                    ##################
                    
                    if env_actions[i]['action']['back_path'] == None:
                        wait_for_move = self.single_turn_control(positions[i],headings[i],env_actions[i]['action']['ghost_pos'])
                    else:
                        wait_for_move = self.teleport_control(positions[i],headings[i],env_actions[i]['action']['ghost_pos'])
                    ##################
                


    def single_turn_control(self, agent_pos, agent_heading, goal_pos):
        ang, _, dis = calculate_vp_rel_pos_fts(agent_pos,goal_pos,agent_heading,0.)
        ang = (ang + 2*math.pi) % (2*math.pi)
        if 0<=ang and ang <  math.pi:
            action_type = 2 # Turn left
            action_value = math.degrees(ang) * 0.27 
        else:
            action_type = 3 # Turn right
            action_value = math.degrees(2*math.pi - ang) * 0.27

        action = np.array([action_type,action_value])
        print(action)
        self.action_client.send_data(action)
        return dis

    def single_move_control(self,dis):        
        action_type = 0 # Move forward
        action_value = dis * 60.
        action = np.array([action_type,action_value])
        print(action)
        self.action_client.send_data(action)
        return None


    def teleport_control(self, agent_pos, agent_heading, goal_pos):
        rel_heading, rel_elevation, _ = calculate_vp_rel_pos_fts(agent_pos,goal_pos,agent_heading,0.)
        action = np.array([-1.,goal_pos[0].item(),goal_pos[2].item(),(rel_heading + agent_heading)%(2*math.pi)])
        print(action)
        self.action_client.send_data(action)
        return None

    def no_action_control(self):
        self.action_client.send_data(np.array([-100.,-100.]))

    def turn_for_observe(self):
        self.action_client.send_data(np.array([3,12.]))