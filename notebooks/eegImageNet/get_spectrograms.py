# Imports
import mne
from types import SimpleNamespace

from utils import load_MNE, load_data, get_loaders, get_args, get_mne_info

import matplotlib.pyplot as plt


for subject in range(0, 8):
    loaders = get_loaders( get_args(subject=subject, spectrograms = True) )
