import os
import sys
import numpy as np
import torch
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Path(object):
    @staticmethod
    def db_dir(database):
        if database == "best_model_dataset":
            root_dir = "./dataset/dur0.1_dis0"
            output_dir = "./dataset/dur0.1_dis0"
            return root_dir, output_dir
        
        elif database == "fast_model_dataset":
            root_dir = "./dataset/dur0.1_dis10"
            output_dir = "./dataset/dur0.1_dis10"

            return root_dir, output_dir

        elif database == "dur0.2_dis21":
            root_dir = "./dataset/dur0.2_dis21"
            output_dir = "./dataset/dur0.2_dis21"
            return root_dir, output_dir
        
        else:
            print("Database {} not available".format(database))
            raise NotImplementedError

class VideoDataset(Dataset):
    def __init__(self, dataset = "fast_model_dataset", split = "test", clip_len = 16, preprocess = False, augmentation : bool = True):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.augmentation = augmentation

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if preprocess:
            if not os.path.exists(os.path.join("./dataset/", 'dataloaders')):
                os.mkdir(os.path.join("./dataset/", 'dataloaders'))
                
            with open('./dataset/dataloaders/{}.txt'.format(dataset), 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])

        if buffer.shape[0] < self.clip_len :
            buffer = self.load_frames(self.fnames[index-(self.clip_len - buffer.shape[0])])

        if buffer.shape[0] < self.clip_len:
            buffer = self.refill_temporal_slide(buffer)

        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == "train" and self.augmentation:
            buffer = self.brightness(buffer, val = 30, p = 0.25)
            buffer = self.contrast(buffer, 1, 1.5, p = 0.25)
            buffer = self.blur(buffer, p = 0.25, kernel_size = 5)
            buffer = self.randomflip(buffer, p = 0.25)
            buffer = self.vertical_shift(buffer, ratio = 0.2, p = 0.25)
            buffer = self.horizontal_shift(buffer, ratio = 0.2, p = 0.25)

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def refill_temporal_slide(self, buffer):
        # if temporal length of buffer is not enought to clip len due to data leakage
        # copy some nearby data 

        for _ in range(self.clip_len - buffer.shape[0]):
            frame_new = buffer[-1].reshape(1, self.resize_height, self.resize_width, 3)
            buffer = np.concatenate((buffer, frame_new))
            
        return buffer

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        if not os.path.exists(os.path.join(self.output_dir, 'train')):
            os.mkdir(os.path.join(self.output_dir, 'train'))

        if not os.path.exists(os.path.join(self.output_dir, 'val')):
            os.mkdir(os.path.join(self.output_dir, 'val'))

        if not os.path.exists(os.path.join(self.output_dir, 'test')):
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            
            if file not in ["disruption", "normal"]:
                continue

            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]
            
            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in tqdm(train):
                self.process_video(video, file, train_dir)

            for video in tqdm(val):
                self.process_video(video, file, val_dir)

            for video in tqdm(test):
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        count = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                frame = np.zeros((self.resize_width, self.resize_height))

            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(count))), img=frame)
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer, p :float = 0.5):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < p:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def horizontal_shift(self, buffer, ratio : float = 0.0, p : float = 0.5):
        if np.random.random() < p:
            ratio = random.uniform(-ratio, ratio)
            to_shift = int(self.crop_size * ratio)
            if ratio > 0:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:,:-to_shift, :] = frame[:,:-to_shift, :]
                    buffer[i] = ref

                    #frame = frame[:,:-to_shift, :]
                    #buffer[i] = cv2.resize(frame, dsize = (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            else:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:,-to_shift:, :] = frame[:,-to_shift:, :]
                    buffer[i] = ref

                    #frame = frame[:,-to_shift:, :]
                    #buffer[i] = cv2.resize(frame, dsize = (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)

        return buffer

    def vertical_shift(self, buffer, ratio : float = 0.0, p : float = 0.5):
        if np.random.random() < p:
            ratio = random.uniform(-ratio, ratio)
            to_shift = int(self.crop_size * ratio)
            if ratio > 0:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[:-to_shift, :, :] = frame[:-to_shift, :, :]
                    buffer[i] = ref
                    #frame = frame[:-to_shift, :, :]
                    #buffer[i] = cv2.resize(frame, dsize = (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                    
            else:
                for i, frame in enumerate(buffer):
                    ref = np.zeros_like(frame)
                    ref[-to_shift:, :, :] = frame[-to_shift:, :, :]
                    buffer[i] = ref

                    #frame = frame[-to_shift:, :, :]
                    #buffer[i] = cv2.resize(frame, dsize = (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
        return buffer

    def blur(self, buffer, p : float = 0.5, kernel_size : int = 5):
        if np.random.random() < p:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def brightness(self, buffer, val : int = 30, p : float = 0.5):
        bright = int(random.uniform(-val, val))
        if np.random.random() < p:
            if bright > 0:
                for i, frame in enumerate(buffer):
                    frame = buffer[i] + bright
                    buffer[i] = np.clip(frame, 10, 255)
            else:
                for i, frame in enumerate(buffer):
                    frame = buffer[i] - bright
                    buffer[i] = cv2.flip(frame, flipCode=1)
            return buffer
        else:
            return buffer

    def contrast(self, buffer, min_val : float = 1.0, max_val : float = 1.5, p : float = 0.5):
        if np.random.random() < p:
            alpha = int(random.uniform(min_val, max_val))
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.convertScaleAbs(frame, alpha = alpha)
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        if buffer.shape[0] < clip_len :
            time_index = np.random.randint(abs(buffer.shape[0] - clip_len))
        elif buffer.shape[0] == clip_len :
            time_index = 0
        else :
            # print("buffer.shape[0] : ", buffer.shape[0])
            time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


if __name__ == "__main__":
    # dataset = 'best_model_dataset'
    dataset = 'fast_model_dataset'

    test_data = VideoDataset(dataset=dataset, split='test', clip_len=8)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(test_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break