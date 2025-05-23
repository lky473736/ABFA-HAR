#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import random
import tensorflow as tf

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset paths
DATASET_PATHS = {
    "UCI_HAR": os.path.join(DATA_DIR, "UCI_HAR_Dataset"),
    "WISDM": os.path.join(DATA_DIR, "WISDM_ar_v1.1"),
    "mHealth": os.path.join(DATA_DIR, "MHEALTHDATASET"),
    "PAMAP2": os.path.join(DATA_DIR, "PAMAP2_Dataset")
}

# Global training parameters
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
PATIENCE = 10

# ABFA Model architecture parameters (Full Model: 1+2+3+4+5+6)
# 1: Initial Projection
INITIAL_PROJECTION_FILTERS = [64, 128]

# 2: MKTC
MKTC_FILTERS = 128
MKTC_KERNEL_SIZES = [3, 7, 11]

# 3: ABFA Block
ABFA_FILTERS = 128
ABFA_DROPOUT_RATE = 0.2

# 4: MSA Block
MSA_FILTERS = 128
MSA_KERNEL_SIZES = [3, 5, 7]

# 5: Transformer Encoder Block
TRANSFORMER_UNITS = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_KEY_DIM = 64

# 6: Classification Head
CLASSIFICATION_UNITS = [128, 64]

# Dropout and regularization
DROPOUT_RATE = 0.2
BN_MOMENTUM = 0.99
LAYER_NORM_EPSILON = 1e-6

# Learning rate scheduling
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 5
MIN_LR = 1e-5

# Dataset-specific window parameters
WINDOW_PARAMS = {
    "UCI_HAR": {"window_size": 128, "step": 64, "sampling_rate": 50, "num_classes": 6},
    "WISDM": {"window_size": 80, "step": 40, "sampling_rate": 20, "num_classes": 6},
    "mHealth": {"window_size": 50, "step": 25, "sampling_rate": 50, "num_classes": 12},
    "PAMAP2": {"window_size": 100, "step": 50, "sampling_rate": 100, "num_classes": 12}
}

# Dataset-specific class information
DATASET_CLASSES = {
    "UCI_HAR": {
        0: "WALKING",
        1: "WALKING_UPSTAIRS", 
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING"
    },
    "WISDM": {
        0: "Walking",
        1: "Jogging", 
        2: "Sitting",
        3: "Standing",
        4: "Upstairs",
        5: "Downstairs"
    },
    "mHealth": {
        0: "Standing still",
        1: "Sitting and relaxing",
        2: "Lying down",
        3: "Walking",
        4: "Climbing stairs",
        5: "Waist bends forward",
        6: "Frontal elevation of arms",
        7: "Knees bending (crouching)",
        8: "Cycling",
        9: "Jogging",
        10: "Running",
        11: "Jump front & back"
    },
    "PAMAP2": {
        0: "lying",
        1: "sitting", 
        2: "standing",
        3: "walking",
        4: "running",
        5: "cycling",
        6: "Nordic walking",
        7: "ascending stairs",
        8: "descending stairs",
        9: "vacuum cleaning",
        10: "ironing",
        11: "rope jumping"
    }
}

def set_seed(seed=SEED):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Enable deterministic operations in TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Configure GPU memory growth to avoid memory allocation issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

def get_dataset_config(dataset_name):
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Dataset configuration
    """
    if dataset_name not in WINDOW_PARAMS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = WINDOW_PARAMS[dataset_name].copy()
    config['classes'] = DATASET_CLASSES[dataset_name]
    config['dataset_path'] = DATASET_PATHS[dataset_name]
    
    return config