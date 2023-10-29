import cv2 
import numpy as np

import torch
from loftr.loftr import LoFTR
from loftr.utils.cvpr_ds_config import default_cfg
from loutils import make_query_image, get_coarse_match, make_student_config

import config as cfg



class Lovio():
    
    def __init__(self, 
                 weights="./weights//checkpoints_new_data/LoFTR_39.pt",
                 device='cuda',
                ):
        self.device = device
        
        # Initialize model
        model_cfg = make_student_config(default_cfg)
        matcher = LoFTR(config=model_cfg)
        matcher.load_state_dict(torch.load(weights)['model_state_dict'])
        self.matcher = matcher.eval().to(device=self.device)
        self.img_size = (model_cfg['input_width'], model_cfg['input_height'])
        self.loftr_coarse_resolution = model_cfg['resolution'][0]
        
        self.M = None


    def match(self, p_frame, n_frame):
        
       # match previoius and current images with LoFTR
        p_frame_1 = make_query_image(p_frame.copy(), self.img_size)
        p_frame_torch = torch.from_numpy(p_frame_1)[None][None].to(device=self.device) / 255.0
        n_frame_1 = make_query_image(n_frame.copy(), self.img_size)
        n_frame_torch = torch.from_numpy(n_frame_1)[None][None].to(device=self.device) / 255.0
        
        with torch.no_grad():
            conf_matrix, _ = self.matcher(p_frame_torch, n_frame_torch)
            conf_matrix = conf_matrix.cpu().numpy()

            mkpts0, mkpts1, mconf = get_coarse_match(conf_matrix, 
                                                     self.img_size[1], 
                                                     self.img_size[0], 
                                                     self.loftr_coarse_resolution)

            # filter only the most confident features
            n_top = 12
        
            indices = np.argsort(mconf)[::-1]
            indices = indices[:n_top]
            mkpts0 = mkpts0[indices, :]
            mkpts1 = mkpts1[indices, :]
        
        self.M, self.mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 1.0)
        
    
class Track():
    
    def __init__(self, time0, lat0, lon0, alt0=0):
        
        self.steps = np.asarray([[0, 0]])
        self.step_sizes = np.asarray([0])
        self.track = np.asarray([[0, 0]])
        self.gps = np.asarray([[lat0, lon0, alt0]])
        self.lat0 = lat0
        self.times = np.asarray([time0])
        self.vels = np.asarray([[0, 0]])
        
        
    def is_outlier(self, step_len):
        
        steps_tail = self.step_sizes[-cfg.NUM_OK_STEPS-1:].copy()
        
        # sheck if we have enough points to detect outlier
        if len(steps_tail)<cfg.NUM_OK_STEPS:
            
            return False
        
        else:
            
            std = np.std(steps_tail)
            mean = np.mean(steps_tail)
            z_score = (step_len - mean) / std
            
            if z_score > cfg.OUT_THRESH:
                
                return True
            
            else:
                
                return False
            
        
    def update(self, ts, heading, step, alt):
        
        self.times = np.hstack((self.times, ts))
        dt = self.times[-1] - self.times[-2]

        step_len = np.linalg.norm(step)
        
        if not self.is_outlier(step_len):
            
            self.step_sizes = np.hstack((self.step_sizes, step_len))
        
            heading += cfg.YAW_BIAS
            RotM = np.asarray([np.cos(heading), -np.sin(heading), 
                               -np.sin(heading), -np.cos(heading)
                              ]).reshape(2,2)
        
            oriented_step = RotM.dot(step.reshape(-1,2).T).T.reshape(-1,2)
        
        else:
            
            oriented_step = np.asarray([[0,0]])
            
        self.steps = np.vstack((self.steps, oriented_step))
        
        drvect = oriented_step/ cfg.focal* alt
        self.vels = np.vstack((self.vels, -drvect/dt*100))
        
            
        new_position = self.track[-1] + drvect
        self.track = np.vstack((self.track, new_position))
        
        dlat = -new_position[0,1]/111139
        dlon = -new_position[0,0]/111139/np.cos(self.lat0/180*np.pi) 
        
        pos = np.asarray([[dlat, dlon, alt]]) + self.gps[0]
        self.gps = np.vstack((self.gps, pos))