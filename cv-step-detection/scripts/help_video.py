#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:29:42 2023

@author: abdulzaf
"""

import cv2
import os

def read_video(dir_input, video):
    # get video data
    vid_cap = cv2.VideoCapture(os.path.join(dir_input, video))
    vid_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)
    # get video frames
    vid_frames = []
    for i in range(vid_len):
        success, image = vid_cap.read()
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA) 
        if success:
            vid_frames.append(image)

    return vid_frames, vid_len, vid_fps


def export_video(path_out, filename, frames_out, fps, vid_res):
    """
    Parameters
    ----------
    path_out : string
    filename : string
    frames_out : list of video frames
    fps : int
    vid_res : list of video resolution

    Returns
    -------
    None.
    """
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