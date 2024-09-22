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
from tqdm import tqdm

from mne.time_frequency import tfr_multitaper



class EEGImageNetDataset(Dataset):
    # https://github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset/blob/main/src/de_feat_cal.py

    def __init__(self, args, transform=None):
        self.dataset_dir = args.dataset_dir
        self._info = self.get_mne_info()
        self.fs = self._info.get('sfreq') # For scipy spectrograms
        self.args = args
        self.transform = transform

        # Use Image generation or classification task
        self.image_generation_task = args.image_generation_task

        # Load data (either raw or spectrograms)
        loaded = self.load_data()
            
        self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.idx2label = {i: label for i, label in enumerate(self.labels)}
        self.label2class = parse_class_dict(os.path.join(self.args.dataset_dir, "synset_map_en.txt"))

        # Compute and save spectrograms if they don't exist
        if self.args.spectrograms and not os.path.exists(f'{self.args.spectrogram_dir}/spectrograms_nperseg_{self.args.nperseg}.pth'):

            assert self.args.subject == [-1], "Spectrograms can only be computed for all subjects"

            # Updates self.data
            self.compute_spectrograms()
        
        # if spectrograms and means and stds are not provided (train), calculate them
        if args.spectrograms and args.means is None and args.stds is None:
            # DB-scale spectrograms
            spectrograms = np.array([10 * np.log10(i['spectrograms']) for i in loaded['dataset']])
            self.args.means = np.mean(spectrograms, axis=(0), keepdims=True)
            self.args.stds = np.std(spectrograms, axis=(0), keepdims=True)

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

        if self.args.spectrograms:
            # Filter frequencies to only include those <= 100 Hz
            freq_mask = self.freqs <= 100

            # Normalize spectrograms here - save params for train, and reuse them as input within test and val
            feat = ( 10 * np.log10(self.data[index]['spectrograms'][:, freq_mask, :]) - np.squeeze(self.args.means, axis=0)[:, freq_mask, :] ) / np.squeeze(self.args.stds, axis=0)[:, freq_mask, :]
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            feat = eeg_data[:, 40:440]

        return feat, label

    def __len__(self):
        return len(self.data)
    
    def load_data(self):

        print("Loading data for subjects: ", self.args.subject)

        # Load spectrograms if they're already created, if not they'll be created later
        if self.args.spectrograms and os.path.exists(f'{self.args.spectrogram_dir}/spectrograms_nperseg_{self.args.nperseg}.pth'):
            loaded = torch.load(os.path.join(self.args.spectrogram_dir, f"spectrograms_nperseg_{self.args.nperseg}.pth"))
            self.freqs = loaded['freqs']
            self.times = loaded['times']

        else:
            loaded = torch.load(os.path.join(self.args.dataset_dir, "EEG-ImageNet_1.pth"))

            # if file exists: os.path.join(self.args.dataset_dir, "EEG-ImageNet_2.pth")
            if os.path.exists(os.path.join(self.args.dataset_dir, "EEG-ImageNet_2.pth")):
                loaded['dataset'] += torch.load(os.path.join(self.args.dataset_dir, "EEG-ImageNet_2.pth"))['dataset']

        # Filter by subject
        if self.args.subject != [-1]:
            if isinstance(self.args.subject, int):
                self.args.subject = [self.args.subject]

            loaded['dataset'] = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] in self.args.subject]

        # Filter by class label
        if self.args.class_label:
            loaded['dataset'] = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           self.args.class_label in self.label2class[loaded['dataset'][i]['label']]]

        # Filter by Granularity
        if self.args.granularity == 'coarse':
            loaded['dataset'] = [i for i in loaded['dataset'] if i['granularity'] == 'coarse']
        elif self.args.granularity == 'all':
            pass
        else:
            fine_num = int(self.args.granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            loaded['dataset'] = [i for i in loaded['dataset'] if
                         i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]
            
        # Filter by indices (for train/test/val split)
        if self.args.indices:
            loaded['dataset'] = np.take(loaded['dataset'], list(self.args.indices))

        return loaded
    
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

    def compute_spectrograms(self):

        print("Computing spectrograms..!")

        epochs_data = self.load_MNE(self.data).get_data()

        fs = self._info.get('sfreq')  # Sampling rate in Hz
        nperseg = self.args.nperseg  # segment size
        noverlap = nperseg // 2  # overlap size (as recommeendeed by chatgpt - wrt. time / frequency resolution tradeeoff)

        # Loop over each epoch
        for epoch_idx in tqdm(range(epochs_data.shape[0])):
            epoch_spectrograms = []
            
            # Loop over each channel
            for channel_idx in range(epochs_data.shape[1]):
                
                # Compute the spectrogram
                freqs, times, Sxx = spectrogram(epochs_data[epoch_idx, channel_idx, :], 
                                                fs=fs, 
                                                nperseg=nperseg, 
                                                noverlap=noverlap)
                
                # Sxx is the spectrogram matrix for this channel, append it
                epoch_spectrograms.append(Sxx)
            
            # Update data to include spectrograms
            self.data[epoch_idx]['spectrograms'] = np.array(epoch_spectrograms)
            # delete EEG data as not needed anymore
            del self.data[epoch_idx]['eeg_data']

        # save as torch file
        torch.save({'dataset': self.data, 
                    'labels': self.labels, 
                    'images': self.images,
                    'freqs': freqs,
                    'times': times}, 
                    # save directory
                    os.path.join(self.args.spectrogram_dir, f"spectrograms_nperseg_{self.args.nperseg}.pth"))
        
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
        
        # check if all labels are within dataset
        labels2remove = []
        for label in label2id.keys():
            if label not in labels:
                print(f"Label {label} not in dataset")
                labels2remove.append(label)
                # remove from label2id

        for label in labels2remove:
            del label2id[label]

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
def get_dataset(args):

    eeg_data = EEGImageNetDataset(args)
    # translates numbered index to label
    eeg_data.idx2label = {i: label for i, label in enumerate(eeg_data.labels)}
    # translates label to class - i.e. panda
    eeg_data.label2class = parse_class_dict(os.path.join(args.dataset_dir, "synset_map_en.txt"))

    return eeg_data


def get_loaders(args):

    assert len(args.split) in [2, 3], "train/test/val split should be provided, or two test/val subjects for leave-one-out"

    assert args.means is None and args.stds is None, "Means and Stds should not be provided but only calculated on train"

    assert len(args.split) == 3 or args.subject == [-1], "If Subject split is provided, subject should be -1"

    # Splitting based on percentages
    if sum(args.split) == 1 and len(args.split) == 3:
        
        # Load all data
        data = get_dataset(args)

        # get train, test and val indices, to calculate mean and std on train
        train_indices, test_indices, val_indices = random_split(range(len(data)),
                                                                args.split, 
                                                                generator=torch.Generator().manual_seed(args.seed))
        
        # Load data per group
        args.indices = list(train_indices)
        print("Loading Training data")
        train_dataset = get_dataset(args)

        # Save means and stds for normalization
        args.means, args.stds = train_dataset.args.means, train_dataset.args.stds

        args.indices = list(test_indices)
        print("Loading Testing data")
        test_dataset = get_dataset(args)

        args.indices = list(val_indices)
        print("Loading Validation data")
        val_dataset = get_dataset(args)


    # Splitting based on subjects
    # if len(args.split) == 2 and elements can be evaluated as integers .is_integer()
    elif all(float(i).is_integer() for i in args.split):

        if args.spectrograms and not os.path.exists(f'{args.spectrogram_dir}/spectrograms_nperseg_{args.nperseg}.pth'):
            print("Spectrograms are not computed yet, these will be computed on the full dataset")
            
            # If spectrograms don't exist, they'll be computed by loading the full dataset
            data = get_dataset(args)
            del data

        all_subjects = list(range(0,16))
        # Load data per group
        args.subject = [subj for subj in all_subjects if subj not in args.split]
        print("Loading Training data for subjects: ", args.subject)
        train_dataset = get_dataset(args)

        # Save means and stds for normalization
        args.means, args.stds = train_dataset.args.means, train_dataset.args.stds

        args.subject = args.split[0]
        print("Loading Validation data for subjects: ", args.subject)
        val_dataset = get_dataset(args)

        args.subject = args.split[1]
        print("Loading Testing data for subjects: ", args.subject)
        test_dataset = get_dataset(args)
    
    # Load into DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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

    raw = mne.EpochsArray(data, mne_info, events = np.array(events), tmin=0.04)

    # Set EEG cap montage - using 10/05 system, as CB1 and CB2 are not in the 10/20 system (https://mne.discourse.group/t/to-set-my-montage-correctly/7202/2)
    template_1005_montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(template_1005_montage, match_case=False, match_alias=True)

    return raw

# Args wrapper, for easy adaptability
def get_args(dataset_dir = 'data/raw/EEG-ImageNet', 
             spectrogram_dir = 'data/processed/EEG-ImageNet', 
             figure_dir = 'figures/analysis/EEG-ImageNet',
             subject = -1, 
             granularity = 'all', 
             image_generation_task = False, 
             class_label = None, 
             split = [0.7, 0.15, 0.15], 
             # if sum to 1 then split percentage-wise - if lists of ints then split by subjects. i.e. [[0,1,2,3,4,5],[6],[7]]
             batch_size = 32, 
             seed = 100, 
             spectrograms = False, 
             freq_bands = {
                            "delta": [0.5, 4],
                            "theta": [4, 8],
                            "alpha": [8, 13],
                            "beta": [13, 30],
                            "gamma": [30, 80]
                        },
            means = None,
            stds = None,
            indices = None,
            nperseg = 200,
            nfft = 256
            ):

    args = SimpleNamespace(**{
        'dataset_dir': dataset_dir, 
        'spectrogram_dir': spectrogram_dir,
        'figure_dir': figure_dir,
        'subject': subject,
        'granularity': granularity,
        # generate images or classify?
        'image_generation_task': image_generation_task,
        'class_label': class_label,
        'split': split,
        'batch_size': batch_size,
        'seed': seed,
        'spectrograms': spectrograms,
        'freq_bands': freq_bands,
        # For standardization of spectrograms
        'means': means,
        'stds': stds,
        # For train/test/val split - if not provided, will use all data
        'indices': indices,
        # For spectrogram computation
        'nperseg': nperseg,
        'nfft': nfft
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

# Create correct shape for SVM
def prepare_data(loader):
    X = []
    y = []
    
    for data, labels in tqdm(loader):
        # Flatten the data from (62, 21, 3) to (62 * 21 * 3)
        data = data.view(data.shape[0], -1).numpy()  # (batch_size, 62*21*3)
        labels = labels.numpy()  # Convert labels to numpy
        
        X.append(data)
        y.append(labels)
    
    # Concatenate all batches into single arrays
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    return X, y