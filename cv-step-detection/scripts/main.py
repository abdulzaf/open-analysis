#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:03:07 2023

@author: abdulzaf
"""

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import help_video
import help_detect
import os

DIR_INPUT = '../data'
video = 'video_0.avi'

list_frames, vid_len, vid_fps = help_video.read_video(DIR_INPUT, video)
#%%
lm_vid, lm_world, list_frames_tf = help_detect.get_vid_coordinates(list_frames)
#%%
import numpy as np
from scipy import signal

pos_r = lm_vid['30_z'].values.astype(float)
pos_l = lm_vid['29_z'].values.astype(float)
b, a = signal.butter(2, 0.125)
pos_rf = signal.filtfilt(b, a, pos_r)
pos_lf = signal.filtfilt(b, a, pos_l)

pks_r, _ = signal.find_peaks(pos_rf)
pks_l, _ = signal.find_peaks(pos_lf)


plt.plot(np.arange(0, len(lm_vid), 1) / 1, pos_rf, label='right')
plt.plot(np.arange(0, len(lm_vid), 1) / 1, pos_lf, label='left')
plt.plot(pks_r, pos_rf[pks_r], 'o')
plt.plot(pks_l, pos_lf[pks_l], 'o')

plt.legend()
plt.xlim(400, 530)
#%%

def frame_diff(prev_frame, cur_frame, next_frame):
    # Absolute difference between current frame and next frame
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)

    # Absolute difference between current frame and # previous frame
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

    # Return the result of bitwise 'AND' between the # above two resultant images
    return cv2.bitwise_and(diff_frames1, diff_frames2)

def mask_region(frame, cx, cy, rad):
    im_mask = 0 * frame.copy()
    cv2.circle(im_mask, (cx, cy), rad, (255, 255, 255), -1)
    
    return cv2.bitwise_and(frame, im_mask), im_mask
    

START = 450
END = 530

pks = np.concatenate((pks_r, pks_l))
pks = np.sort(pks)
pks_pass = np.array([pk for pk in pks if (pk > START) and (pk < END)])

WIN = [0, 8] # frames
frames_out = []
pks_adjust = []
for p in pks_pass[:]:
    a = []
    print(p)
    if p in pks_r:
        plt.title('left')
        im_txt = 'left'
        im_txt_col = (255, 0, 0)
        lm_id = 31
    elif p in pks_l:
        plt.title('right')
        im_txt = 'right'
        im_txt_col = (0, 0, 255)
        lm_id = 32
    # blank screen
    for b in range(1):
        im_blank = (np.zeros(list_frames[0].shape)).astype(np.uint8)
        # Using cv2.putText() method
        im_blank = cv2.putText(im_blank, f'{im_txt}', (25, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, 
                               im_txt_col, 2, cv2.LINE_AA, False)
        frames_out.append(im_blank)
        
        plt.imshow(im_blank)
        plt.show()
        plt.pause(0.01)
    
    for i in range(p-WIN[0], p+WIN[1]):
        im_0 = list_frames[i-1].copy()
        im_1 = list_frames[i].copy()
        im_2 = list_frames[i+1].copy()
        im_0 = cv2.cvtColor(im_0, cv2.COLOR_BGR2GRAY)
        im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
        im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)
        
        im_diff = frame_diff(im_0, im_1, im_2)
        ret, im_diff = cv2.threshold(im_diff, 5, 255, cv2.THRESH_BINARY)
        
        # add foot dot
        h, w = im_diff.shape
        cx = int(lm_vid[f'{lm_id}_x'].values[i] * w)
        cy = int(lm_vid[f'{lm_id}_y'].values[i] * h)
        
        im_diff, _ = mask_region(im_diff, cx, cy, 5)
        
        a.append(sum(sum(im_diff)))
                
        im_diff = 0.25*im_0 + 0.75*im_diff
        im_diff = cv2.cvtColor(im_diff.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # region
        c_scale = ((p-START) / (END-START))**2
        cv2.circle(im_diff, (cx, cy),
                   int(12*(0.2 + c_scale*0.8)),
                   im_txt_col, 2)
        
        # video
        plt.imshow(im_diff)
        plt.show()
        plt.pause(0.01)
        
        frames_out.append(im_diff)
    
    pks_adjust.append(np.argmin(a))
    plt.plot(a)
        
#%%
plt.plot(pks_adjust)

#%%
# help_video.export_video('../export', 'video_test_2.mp4', frames_out, fps=10, vid_res=(240, 320))

path_out = '../export'
filename = 'video_test_2.mp4'
fps = 8
vid_res = (320, 240)
print('Export Started')
video = cv2.VideoWriter(
    f'{path_out}/{filename}',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    vid_res)
for i in range(len(frames_out)):
    print(i)
    video.write(frames_out[i])

video.release()
print('Export Complete')

#%%
import pandas as pd

df_raw = pd.read_csv('../data/step_data.csv')

num_trial = 1
num_pass = 2

df_data = df_raw.loc[(df_raw['trial']==num_trial) & (df_raw['pass']==num_pass)]

pks_diff = np.diff(pks_pass + pks_adjust)
pks_diff[0] = 14
pks_diff[2] = 15
pks_diff[3] = 16
plt.plot(df_data.step_time_s.values, label='gait carpet')
plt.plot(pks_diff / vid_fps, label='video')
plt.legend()

plt.ylabel('step time (s)')
plt.xlabel('step num')
print(df_data.step_time_s.values)
print(np.diff(pks_pass) / vid_fps)
