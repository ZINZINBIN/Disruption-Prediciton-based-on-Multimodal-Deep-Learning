'''
    Generate video data from .api to .png in each folder
    In the original code, we have to split each data as folder
    but it is not efficient since we have to do experiment with different prediction time
    So, we only split the data as folder with respect to the shot number
    - raw_video_path : path for video data 
    - df_shot_list_path : path for dataframe with video shot number and thermal quench time
    - fps : frame per second, default = 210
'''
import cv2, os
import numpy as np
import pandas as pd 
import argparse
from functools import partial
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# parser
parser = argparse.ArgumentParser(description="generate dataset from raw video + numerical data")

# video setting : fps, duration, distance and gap
parser.add_argument("--fps", type = int, default = 210)

# path for data + shot list
parser.add_argument("--raw_video_path", type = str, default = "./dataset/raw_videos/raw_videos/")
parser.add_argument("--df_shot_list_path", type = str, default = "./dataset/KSTAR_Disruption_Shot_List_extend.csv")

# path for saving result(=dataset)
parser.add_argument("--save_path", type = str, default = "./dataset/temp")

# video parameter
parser.add_argument("--width", type = int, default = 256)
parser.add_argument("--height", type = int, default = 256)
parser.add_argument("--overwrite", type = bool, default = True)

args = vars(parser.parse_args())

def frame_calculator(time, fps=210, gap=0):
    frame_time = 1./fps
    frame_num = time/frame_time
    frame_num = frame_num + gap
    return round(frame_num)

def check_directory(save_path : str, shot_num : int):
    # save path 
    save_path = os.path.join(save_path, str(shot_num))
    
    if os.path.isdir(save_path) == False :
        os.mkdir(save_path)

def make_folder(
    shot_num : int,  
    raw_videos_path : str, 
    save_path : str,
    width : int,
    height : int,
    overwrite : bool
    ):
    
    ''' argument
        - raw_videos_path : directory for .avi data
        - save_path : directory for saving .png file
        - shot_num : plasma operation shot number
        - width : resize width
        - height : resize height
    '''
    
    # check the directory for saving video data
    check_directory(save_path, shot_num)

    save_path = os.path.join(save_path, str(shot_num))
    
    # path for video data
    video_shot = "%06dtv01.avi"%shot_num
    video_path = raw_videos_path + video_shot
    is_flip = False

    if not os.path.isfile(video_path):
        video_shot = "%06dtv02.avi"%shot_num
        video_path = raw_videos_path + video_shot
        is_flip = True

    if os.path.isfile(video_path) :
        capture = cv2.VideoCapture(video_path)
        frame_rate = int(round(capture.get(cv2.CAP_PROP_FPS)))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if not overwrite and os.path.isdir(save_path):
            return

        count = 0
        retaining = True

        while(count < frame_count and retaining):
            retaining, frame = capture.read()

            if frame is None:
                continue

            if frame_height != height or frame_width != width:
                frame = cv2.resize(frame, (width, height))
  
            cv2.imwrite(filename = os.path.join(save_path, '%06d.jpg'%count), img = frame)
            count += 1
        
        capture.release()   


if __name__ == "__main__":
    
    fps = args["fps"]
    raw_video_path = args["raw_video_path"]
    video_shot_list_path = args['df_shot_list_path']
    save_path = args["save_path"]
    width = args['width']
    height = args['height']
    overwrite = args['overwrite']

    # video shot list
    video_shot_df = pd.read_csv(video_shot_list_path, encoding = "euc-kr")
    video_shot_df.reset_index(drop = True, inplace = True)
    
    shot_list = video_shot_df.shot.values

    # multi-processing for video - numerical dataset preparation
    n_procs = cpu_count()

    pool = Pool(processes=n_procs)

    make_data_per_proc = partial(
        make_folder, 
        raw_videos_path = raw_video_path, 
        save_path = save_path,
        width = width,
        height = height,
        overwrite = overwrite
    )

    with tqdm(total=len(shot_list)) as pbar:
        for _ in tqdm(pool.imap_unordered(make_data_per_proc, shot_list)):
            pbar.update()

    pool.close()
    pool.join()

    print("######## process done ########")