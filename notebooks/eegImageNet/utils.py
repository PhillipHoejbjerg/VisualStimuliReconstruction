# Imports
import os
import numpy as np
import mne
from types import SimpleNamespace

from scipy.signal import spectrogram
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pickle

# imprt notebook tqdm
from tqdm.notebook import tqdm

from mne.time_frequency import tfr_multitaper



class EEGImageNetDataset(Dataset):
    # https://github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset/blob/main/src/de_feat_cal.py

    def __init__(self, args, transform=None):
        self.dataset_dir = args.dataset_dir
        self._info = self.get_mne_info()
        self.fs = self._info.get('sfreq') # For scipy spectrograms
        self.args = args

        self.transform = transform
        loaded = torch.load(os.path.join(args.dataset_dir, "EEG-ImageNet_1.pth"))
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # translates numbered index to label
        self.idx2label = {i: label for i, label in enumerate(self.labels)}
        # translates label to class - i.e. panda
        self.label2class = parse_class_dict(os.path.join(args.dataset_dir, "synset_map_en.txt"))

        if args.subject != -1:
            loaded['dataset'] = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] == args.subject]
        
        # Filter by class label (only for visualization and analysis)
        if args.class_label:
            loaded['dataset'] = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           args.class_label in self.label2class[loaded['dataset'][i]['label']]]

        chosen_data = loaded['dataset']

        # Some data has more coarse labels than others
        if args.granularity == 'coarse':
            self.data = [i for i in chosen_data if i['granularity'] == 'coarse']
        elif args.granularity == 'all':
            self.data = chosen_data

        else:
            fine_num = int(args.granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [i for i in chosen_data if
                         i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]
            
        # What is this?
        self.spectrograms = args.spectrograms

        if self.spectrograms:
            self.frequency_feat = self.compute_spectrograms(self.data)

        # Use Image as target, or class label
        self.image_generation_task = args.image_generation_task

    def __getitem__(self, index):
        if self.image_generation_task:
            path = self.data[index]["image"]
            label = Image.open(os.path.join(self.dataset_dir, "imageNet_images", path.split('_')[0], path))
            if label.mode == 'L':
                label = label.convert('RGB')
            if self.transform:
                label = self.transform(label)
            else:
                label = path
            # print(f"{index} {path} {label.size()}")
        
        # Classification task
        else:
            label = self.labels.index(self.data[index]["label"])

        if self.spectrograms:
            feat = self.frequency_feat[index]
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            feat = eeg_data[:, 40:440]

        return feat, label

    def __len__(self):
        return len(self.data)
    
    def get_mne_info(self):

        SAMPLING_FREQ = 1000

        # Possibly as input if supposed to work with other datasets
        CH_NAMES = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
                    "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
                    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2",
                    "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]

        # MNE Info object
        _info = mne.create_info(ch_names=CH_NAMES, sfreq=SAMPLING_FREQ, ch_types='eeg')

        return _info

    def compute_spectrograms(self, data):

        # Should be similar to spectrogram, but read up on this
        epochs = self.load_MNE(data)

        # Set window length in seconds (for example, 0.1 seconds)
        win_length = 0.1  # 100 ms window

        # epochs.get_data() get 1000 at a time from data
        # Compute the STFT using multitaper with a single taper (acts as STFT)
        power = tfr_multitaper(epochs, freqs=np.arange(0.5, 100, 0.5), 
                            time_bandwidth=2.0, 
                            n_cycles=win_length * np.arange(0.5, 100, 0.5), 
                            use_fft=True, return_itc=False, decim=3, n_jobs=1, average=False)
            
        # Save power.data in a pickle file
        with open(f'data_subject_{self.args.subject}.pkl', 'wb') as f:
            pickle.dump(power.data, f)

        return power
        
    # Load into MNE
    def load_MNE(self, eeg_data):

        mne_info = get_mne_info()

        data, labels = [], []
        for event in eeg_data:
            data.append(event['eeg_data'][:, 40:440].numpy())  # Convert tensors to numpy arrays
            labels.append(event['label'])

        data, labels = np.array(data), np.array(labels)

        # Events (observation index, 0, label index)
        events = [[i, 0, self.labels.index(label)] for i, label in enumerate(labels)]  # Create events for MNE # later on should probably be changed to specific image - though currently this is just for creating spectrogarms

        label2id = {label: i for i, label in enumerate(self.labels)}
        epochs = mne.EpochsArray(data, mne_info, events = np.array(events), event_id = label2id)

        # Set EEG cap montage - using 10/05 system, as CB1 and CB2 are not in the 10/20 system (https://mne.discourse.group/t/to-set-my-montage-correctly/7202/2)
        template_1005_montage = mne.channels.make_standard_montage('standard_1005')
        epochs.set_montage(template_1005_montage, match_case=False, match_alias=True)

        return epochs



# Read labels
def parse_class_dict(filename):
    result_dict = {}

    with open(filename, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        parts = line.split(maxsplit=1)
        key = parts[0]
        value = parts[1].replace('\n', '').split(', ') if len(parts) > 1 else []
        result_dict[key] = value
    
    return result_dict


# Load into Torch
def load_data(args):

    eeg_data = EEGImageNetDataset(args)
    # translates numbered index to label
    eeg_data.idx2label = {i: label for i, label in enumerate(eeg_data.labels)}
    # translates label to class - i.e. panda
    eeg_data.label2class = parse_class_dict(os.path.join(args.dataset_dir, "synset_map_en.txt"))

    return eeg_data


def get_loaders(args):

    # Load data
    data = load_data(args)

    # Shuffle and split data
    train_dataset, test_dataset, val_loader = random_split(data, 
                                                           args.split, 
                                                           generator=torch.Generator().manual_seed(args.seed))
    
    # Load into DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)

    return {"train": train_loader, "test": test_loader, "val": val_loader}

# Load into MNE
def load_MNE(eeg_data):

    # eeg_data = load_data(args)

    # zero-indexed to remove label
    # access 'eeg_data' if dict
    #mne_data = np.stack([i["eeg_data"].numpy() for i in eeg_data], axis=0) if isinstance(eeg_data[0], dict) else np.stack([i[0].numpy() for i in eeg_data], axis=0) # 4000 x 62 x 501 (i.e. 4000 samples, 62 channels, 501 time points)
    
    mne_info = get_mne_info()

    data, labels = [], []
    for event in eeg_data:
        data.append(event['eeg_data'].numpy())  # Convert tensors to numpy arrays
        labels.append(event['label']) # data['labels'].index(event['label']))  # Convert labels to ints and store them in a list

    data, labels = np.array(data), np.array(labels)

    # Create annotations for MNE
    annotations = []
    onset = 0
    for label, event in zip(labels, data):
        duration = event.shape[1] / mne_info.get('sfreq')  # Calculate the duration of each event
        annotations.append([onset, duration, str(label)])  # Onset, Duration, Label
        onset += duration

    events = [[i, 0, self.labels.index(label)] for i, label in enumerate(labels)]  # Create events for MNE

    raw = mne.EpochsArray(data, mne_info, events = np.array(annotations), tmin=0.04)

    # Set EEG cap montage - using 10/05 system, as CB1 and CB2 are not in the 10/20 system (https://mne.discourse.group/t/to-set-my-montage-correctly/7202/2)
    template_1005_montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(template_1005_montage, match_case=False, match_alias=True)

    return raw

# Args wrapper, for easy adaptability
def get_args(dataset_dir = '../../data/raw/EEG-ImageNet', subject = -1, granularity = 'all', image_generation_task = False, class_label = None, split = [0.7, 0.15, 0.15], batch_size = 32, seed = 100, spectrograms = False, freq_bands = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30],
    "gamma": [30, 80]
}):

    args = SimpleNamespace(**{
        'dataset_dir': dataset_dir, 
        'subject': subject,
        'granularity': granularity,
        # generate images or classify?
        'image_generation_task': image_generation_task,
        'class_label': class_label,
        'split': split,
        'batch_size': batch_size,
        'seed': seed,
        'spectrograms': spectrograms,
        'freq_bands': freq_bands
    })

    return args

def get_mne_info():

    SAMPLING_FREQ = 1000
    CH_NAMES = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
                "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
                "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2",
                "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]

    # MNE Info object
    _info = mne.create_info(ch_names=CH_NAMES, sfreq=SAMPLING_FREQ, ch_types='eeg')

    return _info