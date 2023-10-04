#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:03:07 2023

@author: abdulzaf
"""

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import help_parse
import help_detect
import os

DIR_INPUT = '../data'
video = 'video_0.avi'

list_frames = help_parse.read_video(DIR_INPUT, video)
#%%
START = 450
lm_vid, lm_world, list_frames_tf = help_detect.get_vid_coordinates(list_frames[START:START+100])
#%%
def frame_diff(prev_frame, cur_frame, next_frame):
    # Absolute difference between current frame and next frame
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)

    # Absolute difference between current frame and # previous frame
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

    # Return the result of bitwise 'AND' between the # above two resultant images
    return cv2.bitwise_and(diff_frames1, diff_frames2)

for i in range(25, 100):
    im_0 = list_frames_tf[i-1].copy()
    im_1 = list_frames_tf[i].copy()
    im_2 = list_frames_tf[i+1].copy()
    # im_0 = cv2.cvtColor(im_0, cv2.COLOR_BGR2GRAY)
    # im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
    # im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)
    
    im_diff = frame_diff(im_0, im_1, im_2)
    
    plt.imshow(im_0)
    plt.show()

#%%
plt.plot(lm_vid['31_z'])
plt.plot(lm_vid['32_z'])
