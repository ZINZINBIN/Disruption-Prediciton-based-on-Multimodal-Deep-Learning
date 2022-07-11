from src.dataloader import VideoDataset

if __name__ == "__main__":
    clip_len = 21
    dataset_list = ["dur0.1_dis0", "dur0.1_dis5", "dur0.1_dis10"]

    for dataset in dataset_list:
        train_data= VideoDataset(dataset = dataset, split = "train", clip_len = clip_len, preprocess = True, augmentation=False)
    
    del train_data
