#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:38:02 2023

@author: abdulzaf
"""
import mediapipe as mp
import cv2
import pandas as pd


def get_vid_coordinates(vid_frames):
    # set up limb tracker
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.05)

    # set up pose dict
    lm_idx = [i for i in range(33)]
    lm_props = ['x', 'y', 'z', 'vis']

    lm_dict = {'frame': []}
    lm_world_dict = {'frame': []}
    for idx in lm_idx:
        for prop in lm_props:
            lm_dict[f'{idx}_{prop}'] = []
            lm_world_dict[f'{idx}_{prop}'] = []

    # track pose
    frames_out = []
    for i in range(len(vid_frames)):
        img = vid_frames[i].copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        frame_landmarks = {}
        if results.pose_landmarks:
            # store coordinates
            lm_dict['frame'].append(i)
            lm_world_dict['frame'].append(i)
            for idx in lm_idx:
                lm = results.pose_landmarks.landmark[idx]

                lm_dict[f'{idx}_x'].append(lm.x)
                lm_dict[f'{idx}_y'].append(lm.y)
                lm_dict[f'{idx}_z'].append(lm.z)
                lm_dict[f'{idx}_vis'].append(lm.visibility)
                lm_world_dict[f'{idx}_x'].append(lm.x)
                lm_world_dict[f'{idx}_y'].append(lm.y)
                lm_world_dict[f'{idx}_z'].append(lm.z)
                lm_world_dict[f'{idx}_vis'].append(lm.visibility)
            # draw coordinates
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                frame_landmarks[idx] = [cx, cy]
                # right
                if idx in [24, 26, 28, 30, 32]:
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                # left
                if idx in [23, 25, 27, 29, 31]:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
        frames_out.append(img)

    # compile data
    df_lm = pd.DataFrame.from_dict(lm_dict)
    df_lm_world = pd.DataFrame.from_dict(lm_world_dict)

    return df_lm, df_lm_world, frames_out