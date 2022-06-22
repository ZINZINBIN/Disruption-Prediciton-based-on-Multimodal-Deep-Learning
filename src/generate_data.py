import cv2, os
import numpy as np
import pandas as pd
import itertools
import argparse
from itertools import repeat
from functools import partial
import tqdm
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(description="generate dataset from raw_video data")
parser.add_argument("--fps", type = int, default = 210)
parser.add_argument("--duration", type = float, default = 0.2)
parser.add_argument("--distance", type = int, default = 21)
parser.add_argument("--raw_video_path", type = str, default = "./dataset/raw_videos/raw_videos/")
parser.add_argument("--save_path", type = str, default = "./dataset/")
parser.add_argument("--shot_list_path", type = str, default = "./dataset/KSTAR_Disruption_Shot_List.csv")
parser.add_argument("--gap", type = int, default = 0)

args = vars(parser.parse_args())

fps = args["fps"]
duration = args["duration"]
distance = args["distance"]
raw_video_path = args["raw_video_path"]
save_path = args["save_path"]
gap = args["gap"]
shot_list_path = args["shot_list_path"]

def frame_calculator(time, fps=210, gab=0):
    frame_time = 1./fps
    frame_num = time/frame_time
    frame_num = frame_num+gab
    return round(frame_num)

def new_make_dataset(shot_num, fps, duration, distance, dataset_idx, raw_videos_path, save_path):
    duration_frame = round(fps*duration)
    border_num = fps//duration_frame
    video_shot = "%06dtv01.avi"%shot_num
    video_path = raw_videos_path + video_shot
    
    save_path = save_path + "dur{}_dis{}/".format(duration, distance)
    if os.path.isdir(save_path) == False :
        os.mkdir(save_path)

    dis_path = save_path + "disruption/"
    border_path = save_path + "borderline/"
    nom_path = save_path + "normal/"

    if os.path.isdir(dis_path) == False :
        os.mkdir(dis_path)
    if os.path.isdir(border_path) == False :
        os.mkdir(border_path)
    if os.path.isdir(nom_path) == False :
        os.mkdir(nom_path)
    
    print("############# ", shot_num)
    
    if os.path.isfile(video_path) :
        # load video
        cap = cv2.VideoCapture(video_path)

        frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        name, exe = video_shot.split('.')
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        frame_num = 0
        disruption_bool = False
        save_start = True

        print("(info) dataset_idx : {} - tTQend_frame : {} / duration : {}s({}fps) / distance : {}".format(dataset_idx[shot_num], shot_num, duration, duration_frame , distance))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if dataset_idx[shot_num][1] < frame_num :
                if save_start :
                    save_video = "{}_{}~{}.".format(shot_num, frame_num, frame_num+duration_frame) + exe
                    out = cv2.VideoWriter(nom_path+save_video, fourcc, fps, (w, h))
                    save_start = False
                
                if (frame_num + duration_frame) ==dataset_idx[shot_num][0]:

                    # print("disruption idx: ", frame_num)

                    out.release()
                    save_video = "{}_{}~{}.".format(shot_num,frame_num, frame_num+duration_frame) + exe
                    out = cv2.VideoWriter(dis_path+save_video, fourcc, fps, (w, h))
                    disruption_bool= True
                    
                elif (frame_num + duration_frame * border_num) == dataset_idx[shot_num][0] and border_num != 1:

                    # print("borderline idx : ", frame_num)

                    out.release()
                    save_video = "{}_{}~{}.".format(shot_num, frame_num, frame_num+duration_frame) + exe
                    out = cv2.VideoWriter(border_path+save_video, fourcc, fps, (w, h))
                    border_num -= 1
                    
                elif (frame_num - dataset_idx[shot_num][1])%duration_frame == 0 :
                    if disruption_bool :
                        break

                    # print("nomal idx: ", frame_num)
                    
                    out.release()
                    save_video = "{}_{}~{}.".format(shot_num,frame_num, frame_num+duration_frame) +exe
                    out = cv2.VideoWriter(nom_path+save_video, fourcc, fps, (w, h))

                
                out.write(frame)
            
            frame_num+=1

        try :
            cap.release()
            out.release()
        except Exception as err:
            print("{} err: {}".format(shot_num, err))


if __name__ == "__main__":

    n_procs = cpu_count()

    shot_df = pd.read_csv(shot_list_path, encoding="euc-kr")
    N_index = shot_df["Isdata"][shot_df["Isdata"] == 'N'].index
    shot_df = shot_df.drop(N_index)
    shot_df.reset_index(drop=True, inplace=True)
    shot_df["tTQend_frame"] = shot_df["tTQend"].apply(frame_calculator, gab=gap)
    pre_shot_df = shot_df[["shot", "tTQend_frame"]]

    duration_frame = round(fps*duration)
    dataset_idx = {}

    for shot, end_frame in pre_shot_df.iloc:
        dis_frame = end_frame - distance
        dataset_idx[shot] = [dis_frame, dis_frame%duration_frame]

    pool = Pool(processes=n_procs)
    new_make_data_part = partial(new_make_dataset, 
                            fps=fps, 
                            duration=duration, 
                            distance=distance, 
                            dataset_idx=dataset_idx,
                            raw_videos_path=raw_video_path, 
                            save_path=save_path)
    for i in tqdm.tqdm(pool.imap_unordered(new_make_data_part, pre_shot_df['shot']), total=len(pre_shot_df['shot'])):
        pass
    pool.close()
    pool.join()