#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np
import torch

import vlnce_baselines  # noqa: F401

from transformers import PretrainedConfig
from typing import List
from vlnce_baselines.ss_trainer_ETP import RLTrainer

def main():
    config={
        'SEED':0,
        'Server_IP':'192.168.101.75',
        'Robot_IP':'192.168.101.195',
        'MODEL':{
              'task_type': 'r2r',              
              'pretrained_path': 'pretrained/model_step_100000.pt',
              'fix_lang_embedding': False,
              'fix_pano_embedding': False,
              'use_depth_embedding': True,
              'use_sprels': True,
              'merge_ghost': True,
              'consume_ghost': True,
              'spatial_output': False,
              'RGB_ENCODER':
                {'output_size': 512},   
              'VISUAL_DIM':{
                'vis_hidden': 768,
                'directional': 128},
              'INSTRUCTION_ENCODER':{
                'bidirectional': True},
            },
        'IL':{
              'ckpt_to_load': 'pretrained/ckpt.iter25000.pth',
              'max_traj_len': 50,
              'max_text_len': 80,
              'loc_noise': 0.8,
              'back_algo': 'teleport',
              # 'back_algo': 'control'
            },
        }
    run_exp(config)


def run_exp(config) -> None:

    random.seed(config['SEED'])
    np.random.seed(config['SEED'])
    torch.manual_seed(config['SEED'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    instruction = 'Pass by the table and stop in front of the green plant next to the basketball.'
    trainer = RLTrainer(config)
    trainer._initialize_policy(config=config, load_from_ckpt=True)
    trainer.rollout('eval',instruction)

if __name__ == "__main__":
    main()
