#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:12:16 2023

@author: abdulzaf
"""
import cv2
import os

def read_video(dir_input, video):
    # get video data
    vid_cap = cv2.VideoCapture(os.path.join(dir_input, video))
    vid_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get video frames
    vid_frames = []
    for i in range(vid_len):
        success, image = vid_cap.read()
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA) 
        if success:
            vid_frames.append(image)

    return vid_frames
