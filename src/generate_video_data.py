'''Generate video data code
- raw_video_path : path for video data 
- video_shot_list_path : path for shot list corresponding to video with thermal quench time
- ts_data_path : path for shot list with numerical dataset(=plasma parameter)
- fps : frame per second, default = 210
- distance : time-step posterior to current state
- duration : time-duration used for prediction
- gap : frame gap
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
parser.add_argument("--duration", type = int, default = 21) # duration * fps(=210) = seq_len(or frame length)
parser.add_argument("--distance", type = int, default = 0) # prediction length
parser.add_argument("--gap", type = int, default = int(0.1 * 207))

# path for data + shot list
parser.add_argument("--raw_video_path", type = str, default = "./dataset/raw_videos/raw_videos/")
parser.add_argument("--video_shot_list_path", type = str, default = "./dataset/KSTAR_Disruption_Shot_List_extend.csv")
parser.add_argument("--ts_data_path", type = str, default = "./dataset/KSTAR_Disruption_ts_data_extend.csv")

# path for saving result(=dataset)
parser.add_argument("--save_path", type = str, default = "./dataset/")

args = vars(parser.parse_args())

fps = args["fps"]
duration = args["duration"]
distance = args["distance"]
gap = args["gap"]

raw_video_path = args["raw_video_path"]
video_shot_list_path = args['video_shot_list_path']
ts_data_path = args['ts_data_path']

save_path = args["save_path"]

def select_shot_list(
    video_data : pd.DataFrame,
    ts_data : pd.DataFrame 
    ):

    video_shot_list = video_data.shot.values
    ts_shot_list = np.unique(ts_data.shot.values)

    shot_list = []

    for shot in tqdm(ts_shot_list, desc = "select common shot list between video and numerical dataset"):
        if shot in video_shot_list:
            shot_list.append(shot)
        else:
            pass
    return shot_list

def frame_calculator(time, fps=210, gap=0):
    frame_time = 1./fps
    frame_num = time/frame_time
    frame_num = frame_num + gap
    return round(frame_num)

def check_directory(save_path : str):
    # save path 
    save_path = save_path + "dur{}_dis{}/".format(duration, distance)
    if os.path.isdir(save_path) == False :
        os.mkdir(save_path)

    dis_path = save_path + "disruption/"
    nom_path = save_path + "normal/"

    if os.path.isdir(dis_path) == False :
        os.mkdir(dis_path)
    if os.path.isdir(nom_path) == False :
        os.mkdir(nom_path)

def make_dataset(
    shot_num : int, 
    fps : int, 
    gap : int, 
    duration : int, # duration frame
    distance : int, # distance frame
    raw_videos_path : str, 
    save_path : str,
    video_data : pd.DataFrame, # video shot list with thermal quench time
    ):

    video_df = video_data[video_data.shot == shot_num]

    # thermal quench를 기준으로 disruption 시점을 정의
    tftsrt = video_df['tftsrt'].apply(frame_calculator, fps = fps, gap = gap).values.item()

    # tTQend_frame = video_df['tTQend'].apply(frame_calculator, fps = fps, gap = gap).values.item()
    # dis_frame = tTQend_frame - distance
    tipminf_frame = video_df['frame_tipminf'].values.item()
    cutoff_frame = video_df['frame_cutoff'].values.item()

    dis_frame = tipminf_frame - distance

    # dis_frame에 정수배로 duration 간격에 따라 데이터셋을 구축하기 위해 start_frame을 조정
    start_frame = dis_frame % duration

    save_path = save_path + "dur{}_dis{}/".format(duration, distance)
    dis_path = save_path + "disruption/"
    nom_path = save_path + "normal/"

    video_shot = "%06dtv01.avi"%shot_num
    video_path = raw_videos_path + video_shot
    is_flip = False

    if not os.path.isfile(video_path):
        video_shot = "%06dtv02.avi"%shot_num
        video_path = raw_videos_path + video_shot
        is_flip = True

    if os.path.isfile(video_path) :
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num < tftsrt:
                pass
            else:
                if save_start:
                    save_video = "{}_{}_{}.".format(shot_num, frame_num, frame_num+duration) + exe
                    out = cv2.VideoWriter(nom_path+save_video, fourcc, fps, (w, h))
                    save_start = False
                
                # disruption phase
                if frame_num + duration == dis_frame: 
                    out.release()
                    save_video = "{}_{}_{}.".format(shot_num,frame_num, frame_num+duration) + exe
                    out = cv2.VideoWriter(dis_path+save_video, fourcc, fps, (w, h))
                    disruption_bool= True

                # normal phase
                elif (frame_num - start_frame)%duration == 0 and frame_num != start_frame: 
                    if disruption_bool :
                        break
                    else:
                        out.release()
                        save_video = "{}_{}_{}.".format(shot_num, frame_num, frame_num+duration) +exe
                        out = cv2.VideoWriter(nom_path+save_video, fourcc, fps, (w, h))
                    
                if is_flip:
                    frame = cv2.flip(frame, 1)

                out.write(frame)
                
            frame_num+=1

        try :
            cap.release()
            out.release()
        except Exception as err:
            print("{} err: {}".format(shot_num, err))


if __name__ == "__main__":

    # video shot list
    video_shot_df = pd.read_csv(video_shot_list_path, encoding = "euc-kr")

    # N_index = video_shot_df['Isdata'][video_shot_df['Isdata'] == 'N'].index
    # video_shot_df = video_shot_df.drop(N_index)

    video_shot_df.reset_index(drop = True, inplace = True)

    # shot info list
    ts_shot_df = pd.read_csv(ts_data_path)

    # choose data columns (time series data)
    # ts_cols = ts_shot_df.columns[ts_shot_df.notna().any()].drop(['Unnamed: 0','shot','time']).tolist()

    # check directory
    check_directory(save_path)

    # select shot list
    shot_list = select_shot_list(video_shot_df, ts_shot_df)

    # multi-processing for video - numerical dataset preparation
    n_procs = cpu_count()

    pool = Pool(processes=n_procs)

    make_data_per_proc = partial(
        make_dataset, 
        fps = fps, 
        gap = gap, 
        duration = duration,
        distance = distance, 
        raw_videos_path = raw_video_path, 
        save_path = save_path,
        video_data = video_shot_df,
    )

    with tqdm(total=len(shot_list)) as pbar:
        for _ in tqdm(pool.imap_unordered(make_data_per_proc, shot_list)):
            pbar.update()

    pool.close()
    pool.join()

    print("######## generate data with multiprocessing complete....! ########")