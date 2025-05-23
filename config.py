#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration parameters for the ABFA-MST HAR system.
Action-Prototype Guided Temporal Modeling for Human Activity Recognition
"""

import os
import numpy as np
import random
import tensorflow as tf

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TESTING_DIR = os.path.join(BASE_DIR, "testing")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TESTING_DIR, exist_ok=True)

# Dataset paths
DATASET_PATHS = {
    "PAMAP2": os.path.join(DATA_DIR, "PAMAP2_Dataset"),
    "UCI_HAR": os.path.join(DATA_DIR, "UCI_HAR_Dataset"),
    "WISDM": os.path.join(DATA_DIR, "WISDM_ar_v1.1"),
    "mHealth": os.path.join(DATA_DIR, "MHEALTHDATASET")
}

# Global parameters
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-7
PATIENCE = 10

# Model parameters
EMBEDDING_DIM = 128
TRANSFORMER_NUM_HEADS = 4
DROPOUT_RATE = 0.2
NUM_ACTIVITY_CLASSES = 12  # For ABFA layer

# Window parameters (fixed per dataset)
WINDOW_PARAMS = {
    "PAMAP2": {"window_width": 100, "stride": 50, "sampling_rate": 100},
    "UCI_HAR": {"window_width": 128, "stride": 128, "sampling_rate": 50},
    "WISDM": {"window_width": 20, "stride": 10, "sampling_rate": 20},
    "mHealth": {"window_width": 50, "stride": 25, "sampling_rate": 50}
}

# Class information for UCI_HAR
UCI_HAR_ACTIVITY_LABELS = [
    'WALKING',
    'WALKING_UPSTAIRS',
    'WALKING_DOWNSTAIRS',
    'SITTING',
    'STANDING',
    'LAYING'
]

# Class information for WISDM
WISDM_ACTIVITY_LABELS = [
    'Walking',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Downstairs'
]

# Class information for mHealth
MHEALTH_ACTIVITY_LABELS = [
    'L1: Standing still',
    'L2: Sitting and relaxing',
    'L3: Lying down',
    'L4: Walking',
    'L5: Climbing stairs',
    'L6: Waist bends forward',
    'L7: Frontal elevation of arms',
    'L8: Knees bending (crouching)',
    'L9: Cycling',
    'L10: Jogging',
    'L11: Running',
    'L12: Jump front & back'
]

# Class information for PAMAP2
PAMAP2_ACTIVITY_LABELS = [
    'lying',
    'sitting',
    'standing',
    'walking',
    'running',
    'cycling',
    'Nordic walking',
    'ascending stairs',
    'descending stairs',
    'vacuum cleaning',
    'ironing',
    'rope jumping'
]

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

# ABFA model component configuration
ABFA_COMPONENTS = {
    "INITIAL_PROJECTION": 1,
    "MULTI_SCALE_CNN": 2,
    "ABFA_BLOCK": 3,
    "MST_BLOCK": 4,
    "TRANSFORMER_ENCODER": 5,
    "CLASSIFICATION_HEAD": 6
}

# Full model configuration (all components)
FULL_MODEL_CONFIG = [1, 2, 3, 4, 5, 6]