# Imports
from utils import prepare_data, load_MNE, get_dataset, get_loaders, get_mne_info

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
import mne
import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit, GridSearchCV


# Arguments

parser = argparse.ArgumentParser(description='EEG-ImageNet Argument Parser')

parser.add_argument('--dataset_dir', type=str, default='data/raw/EEG-ImageNet', help='Directory where the dataset is stored.')
parser.add_argument('--spectrogram_dir', type=str, default='data/processed/EEG-ImageNet', help='Directory where the spectrograms are stored.')
parser.add_argument('--figure_dir', type=str, default='figures/analysis/EEG-ImageNet', help='Directory for saving analysis figures.')
parser.add_argument('--results_dir', type=str, default='results/EEG-ImageNet', help='Directory for saving analysis figures.')
parser.add_argument('--subject', type=int, nargs='+', default=[-1], help='List of subject indices. Default is [-1] for all subjects.')
parser.add_argument('--granularity', type=str, default='all',help='Granularity level for processing. Default is "all".')
parser.add_argument('--image_generation_task', action='store_true', help='Flag to indicate if it is an image generation task.')
parser.add_argument('--class_label', type=str, default=None, help='Class label for classification tasks. Default is None.')
parser.add_argument('--split', type=float, nargs='+', default=[0.7, 0.15, 0.15],help='Data split ratios for train/val/test. Default is [0.7, 0.15, 0.15].')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data processing. Default is 32.')
parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility. Default is 100.')
parser.add_argument('--spectrograms', action='store_true', help='Flag to indicate whether to use spectrograms.')
parser.add_argument('--freq_bands', type=dict, default={
                        "delta": [0.5, 4],
                        "theta": [4, 8],
                        "alpha": [8, 13],
                        "beta": [13, 30],
                        "gamma": [30, 80]
                    }, 
                    help='Frequency bands for EEG analysis.')
parser.add_argument('--means', type=float, nargs='+', default=None, help='Mean values for spectrogram standardization.')
parser.add_argument('--stds', type=float, nargs='+', default=None, help='Standard deviation values for spectrogram standardization.')
parser.add_argument('--indices', type=int, nargs='+', default=None, help='Indices for train/val/test split. If not provided, all data is used.')
parser.add_argument('--nperseg', type=int, default=200, help='nperseg parameter for spectrogram computation.')
parser.add_argument('--nfft', type=int, default=256, help='nfft parameter for spectrogram computation.')
# num_workers
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading. Default is 0.')
args = parser.parse_args()

# python notebooks/eegImageNet/train_SVM.py --dataset_dir='data/raw/EEG-ImageNet' --spectrogram_dir='data/processed/EEG-ImageNet' --subject=[1] --num_workers=8 --spectrograms --nperseg=100 --nfft=128

for subject_idx in tqdm(args.subject):
    
    # Splitting based on subject 1
    loaders = get_loaders(args)

    # Print shapes of datasets
    #print(f"Train: {loaders['train'].dataset.data.shape} X {loaders['train'].dataset.data[0]['spectrograms'].shape}")
    #print(f"Val: {loaders['val'].dataset.data.shape} X {loaders['val'].dataset.data[0]['spectrograms'].shape}")
    #print(f"Test: {loaders['test'].dataset.data.shape} X {loaders['test'].dataset.data[0]['spectrograms'].shape}")

    if subject_idx == -1:
        subject_idx = 'all'
        
    # Prepare training, validation, and test data
    X_train, y_train = prepare_data(loaders['train'])
    X_val, y_val = prepare_data(loaders['val'])
    X_test, y_test = prepare_data(loaders['test'])

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Val: {X_val.shape}, {y_val.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    # Combine training and validation sets
    X_combined = np.vstack((X_train, X_val))  # Stack the data vertically
    y_combined = np.hstack((y_train, y_val))  # Stack the labels horizontally

    # Create an index that separates the training and validation sets
    # -1 for training samples, 0 for validation samples
    split_index = [-1] * len(X_train) + [0] * len(X_val)

    # Use PredefinedSplit to specify the training and validation data
    ps = PredefinedSplit(test_fold=split_index)

    # Define the parameter grid for the RBF kernel
    param_grid_rbf = {
        'C': [0.1, 1, 10, 100, 1000],  # Regularization parameter
        'gamma': [1e-3, 1e-2, 1e-1, 'scale', 'auto'],  # Kernel coefficient for RBF
    }

    # Define the parameter grid for the Linear kernel
    param_grid_linear = {
        'C': [0.1, 1, 10, 100, 1000],  # Regularization parameter for Linear kernel
    }

    # Instantiate the SVM models
    svm_rbf = SVC(kernel='rbf')
    svm_linear = SVC(kernel='linear')

    # Perform grid search for the RBF kernel
    grid_search_rbf = GridSearchCV(estimator=svm_rbf, param_grid=param_grid_rbf, cv=ps, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_rbf.fit(X_combined, y_combined)  # Fit on your training data
    # Extract the results into a DataFrame and save to a CSV file
    results_df = pd.DataFrame(grid_search_rbf.cv_results_)
    os.makedirs(f'{args.results_dir}/grid_search_results/', exist_ok=True)
    results_df.to_csv(f'{args.results_dir}/grid_search_results/svm_rbf_grid_search_results_subject_{subject_idx}.csv', index=False)
    # Get the best parameters and scores
    print(f"Best parameters for RBF kernel: {grid_search_rbf.best_params_}")
    print(f"Best accuracy for RBF kernel: {grid_search_rbf.best_score_}")

    # Perform grid search for the Linear kernel
    grid_search_linear = GridSearchCV(estimator=svm_linear, param_grid=param_grid_linear, cv=ps, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_linear.fit(X_combined, y_combined)  # Fit on your training data
    # Extract the results into a DataFrame and save to a CSV file
    results_df = pd.DataFrame(grid_search_linear.cv_results_)
    results_df.to_csv(f'{args.results_dir}/grid_search_results/svm_linear_grid_search_results_subject_{subject_idx}.csv', index=False)
    # Get the best parameters and scores
    print(f"Best parameters for Linear kernel: {grid_search_linear.best_params_}")
    print(f"Best accuracy for Linear kernel: {grid_search_linear.best_score_}")


    # Re-fit best model (depending on which kernel was best)
    best_svm = grid_search_rbf.best_estimator_ if grid_search_rbf.best_score_ > grid_search_linear.best_score_ else grid_search_linear.best_estimator_

    # Validate the retrained model on the validation data
    y_val_pred = best_svm.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy with tuned SVM: {val_accuracy * 100:.2f}%')

    # Test the model on the test data
    y_test_pred = best_svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy with tuned SVM: {test_accuracy * 100:.2f}%')

    # save params
    with open(f'{args.results_dir}/best_params_subject_{subject_idx}.txt', 'w') as f:
        f.write(f"Best parameters for RBF kernel: {grid_search_rbf.best_params_}\n")
        f.write(f"Best accuracy for RBF kernel: {grid_search_rbf.best_score_}\n")
        f.write(f"Best parameters for Linear kernel: {grid_search_linear.best_params_}\n")
        f.write(f"Best accuracy for Linear kernel: {grid_search_linear.best_score_}\n")
        f.write(f'Validation Accuracy with tuned SVM: {val_accuracy * 100:.2f}%\n')
        f.write(f'Test Accuracy with tuned SVM: {test_accuracy * 100:.2f}%')

    # Confusion matrix for test data
    conf_matrix = confusion_matrix(y_test, y_test_pred, normalize='true')

    # Get class labels (assuming label2class mapping is available)
    class_labels = [loaders['train'].dataset.label2class[label][0] for label in loaders['train'].dataset.idx2label.values()]

    # Plot the confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)

    # Rotate x-axis labels vertically and reduce font size
    plt.xticks(rotation=90, fontsize=12)  # Adjust fontsize for x-axis
    plt.yticks(rotation=0, fontsize=12)   # Adjust fontsize for y-axis

    plt.title(f'Subject {subject_idx} - Test Data - Acc: %.2f%%' % (test_accuracy * 100))
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    # create directory, then savefig: {args.figure_dir}/confusion_matrices/    
    os.makedirs(f'{args.figure_dir}/confusion_matrices/', exist_ok=True)
    plt.savefig(f'{args.figure_dir}/confusion_matrices/confusion_matrix_subject_{subject_idx}_optimized.png')


