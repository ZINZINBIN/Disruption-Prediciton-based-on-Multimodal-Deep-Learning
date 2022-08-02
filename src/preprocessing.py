''' 
    Preprocessing Video and Numerical dataset
'''
import cv2, os, glob2
import numpy as np
import pandas as pd
import argparse
from functools import partial
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from typing import List
from sklearn.model_selection import train_test_split

# parser
parser = argparse.ArgumentParser(description="Data preprocessing : convert multi-modal dataset from video and numerical data")

# train-test split ratio
parser.add_argument("--test_ratio", type = float, default = 0.2)
parser.add_argument("--valid_ratio", type = float, default = 0.2)

# data directory
parser.add_argument("--video_data_path", type = str, default = "./dataset/dur21_dis0")
parser.add_argument("--ts_data_path", type = str, default = "./dataset/KSTAR_Disruption_ts_data_extend.csv")
parser.add_argument("--save_path", type = str, default = "./dataset/dur21_dis0")

# image size
parser.add_argument("--resize_height", type = int, default = 256)
parser.add_argument("--resize_width", type = int, default = 256)

args = vars(parser.parse_args())

test_ratio = args["test_ratio"]
valid_ratio = args["valid_ratio"] / (1.0 - test_ratio)
train_ratio = 1.0 - valid_ratio - test_ratio

video_data_path = args['video_data_path']
ts_data_path = args['ts_data_path']
save_path = args['save_path']

height = args['resize_height']
width = args['resize_width']

phase_list = ["disruption", "normal", "borderline"]

def check_directory(save_path : str):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'train'))
        os.mkdir(os.path.join(save_path, 'val'))
        os.mkdir(os.path.join(save_path, 'test'))
    
    if not os.path.exists(os.path.join(save_path, 'train')):
        os.mkdir(os.path.join(save_path, 'train'))
        os.mkdir(os.path.join(save_path, 'train', 'disruption'))
        os.mkdir(os.path.join(save_path, 'train', 'normal'))
    
    if not os.path.exists(os.path.join(save_path, 'valid')):
        os.mkdir(os.path.join(save_path, 'valid'))
        os.mkdir(os.path.join(save_path, 'valid', 'disruption'))
        os.mkdir(os.path.join(save_path, 'valid', 'normal'))
    
    if not os.path.exists(os.path.join(save_path, 'test')):
        os.mkdir(os.path.join(save_path, 'test'))
        os.mkdir(os.path.join(save_path, 'test', 'disruption'))
        os.mkdir(os.path.join(save_path, 'test', 'normal'))

    return

def get_shot_path(video_data_path : str):
    path_disrupt = glob2.glob(video_data_path + "/disruption/*.avi")
    path_normal = glob2.glob(video_data_path + "/normal/*.avi")
    return path_normal, path_disrupt

def video2img(video_path : str, width : int, height : int, save_dir : str):

        capture = cv2.VideoCapture(video_path)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        count = 0
        retaining = True

        if not os.path.exists(os.path.join(save_dir)):
            os.mkdir(os.path.join(save_dir))

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                frame = np.zeros((width, height))

            if (frame_height != height) or (frame_width != width):
                frame = cv2.resize(frame, (width, height))
                
            cv2.imwrite(filename=os.path.join(save_dir, '%04d.jpg'%count), img=frame)
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()
    
def preprocess_per_proc(
    n_proc : int,
    n_procs : int,
    path_normal : List,
    path_disrupt : List,
    valid_ratio : float,
    test_ratio : float,
    height : int,
    width : int, 
    save_path : str
    ):

    normal_start_idx = int(n_proc * len(path_normal) / n_procs)
    normal_end_idx = int((n_proc + 1) * len(path_normal) / n_procs)

    disrupt_start_idx = int(n_proc * len(path_disrupt) / n_procs)
    disrupt_end_idx = int((n_proc + 1) * len(path_disrupt) / n_procs)

    if n_proc == n_procs - 1:
        normal_end_idx = len(path_normal)
        disrupt_end_idx = len(path_disrupt)

    path_normal = path_normal[normal_start_idx : normal_end_idx]
    path_disrupt = path_disrupt[disrupt_start_idx : disrupt_end_idx]

    for phase, path_list in zip(['normal', 'disruption'],[path_normal, path_disrupt]):
        train_and_valid, test = train_test_split(path_list, test_size = test_ratio, random_state=42)
        train, valid = train_test_split(train_and_valid, test_size = valid_ratio, random_state=42)

        train_dir = os.path.join(save_path, 'train', phase)
        valid_dir = os.path.join(save_path, 'valid', phase)
        test_dir = os.path.join(save_path, 'test', phase)

        # example of video_path : dur21_dis0/disruption/{shot_num}_{frame_srt}_{frame_end}.avi
        for video_path in train:
            video_filename = video_path.split('/')[-1].split('.')[0]
            save_location = os.path.join(train_dir, video_filename)
            video2img(video_path, width, height, save_location)
        
        for video_path in valid:
            video_filename = video_path.split('/')[-1].split('.')[0]
            save_location = os.path.join(valid_dir, video_filename)
            video2img(video_path, width, height, save_location)
        
        for video_path in test:
            video_filename = video_path.split('/')[-1].split('.')[0]
            save_location = os.path.join(test_dir, video_filename)
            video2img(video_path, width, height, save_location)


if __name__ == "__main__":

    # check directory
    check_directory(save_path)

    normal, disrupt = get_shot_path(video_data_path)

    # multi-processing for video - numerical dataset preparation
    n_procs = cpu_count()

    pool = Pool(processes=n_procs)

    n_proc_list = [n_proc for n_proc in range(n_procs)]

    preprocess = partial(
        preprocess_per_proc,
        n_procs = n_procs,
        path_normal = normal,
        path_disrupt = disrupt,
        valid_ratio = valid_ratio,
        test_ratio = test_ratio,
        height = height,
        width = width, 
        save_path = save_path
    )

    with tqdm(total=len(n_proc_list)) as pbar:
        for _ in tqdm(pool.imap_unordered(preprocess, n_proc_list)):
            pbar.update()

    pool.close()
    pool.join()

    print("######## preprocessing data complete ....! ########")