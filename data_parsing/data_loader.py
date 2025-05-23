#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import get_dataset_config

def split_sequences(data, labels, window_size, step):
    """
    Split data into sequences with overlapping windows
    
    Args:
        data (numpy.ndarray): Input data
        labels (numpy.ndarray): Labels for each timestamp
        window_size (int): Size of each window
        step (int): Step size between windows
        
    Returns:
        tuple: (X_sequences, y_sequences)
    """
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, step):
        X.append(data[i:i + window_size])
        # Use the label from the last timestamp in the window
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

def load_uci_har_data(data_path, window_size=128, step=64):
    """
    Load UCI HAR dataset
    
    Args:
        data_path (str): Path to UCI HAR dataset
        window_size (int): Window size for sequences
        step (int): Step size for overlapping windows
        
    Returns:
        tuple: (X_sequences, y_sequences, y_onehot, activity_labels)
    """
    print("Loading UCI HAR dataset...")
    
    # Load training data
    train_path = os.path.join(data_path, 'train', 'Inertial Signals')
    test_path = os.path.join(data_path, 'test', 'Inertial Signals')
    
    # Load inertial signals
    train_signals = []
    test_signals = []
    
    signal_files = [
        'body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
        'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
        'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt'
    ]
    
    for signal_file in signal_files:
        train_signals.append(np.loadtxt(os.path.join(train_path, signal_file)))
        test_signals.append(np.loadtxt(os.path.join(test_path, signal_file.replace('train', 'test'))))
    
    # Stack signals
    X_train = np.transpose(np.array(train_signals), (1, 2, 0))
    X_test = np.transpose(np.array(test_signals), (1, 2, 0))
    
    # Load labels
    y_train = np.loadtxt(os.path.join(data_path, 'train', 'y_train.txt')) - 1  # Convert to 0-based
    y_test = np.loadtxt(os.path.join(data_path, 'test', 'y_test.txt')) - 1
    
    # Combine train and test data
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    
    # Flatten for sequence processing if needed
    if X_combined.shape[1] != window_size:
        X_flat = X_combined.reshape(-1, X_combined.shape[2])
        y_flat = np.repeat(y_combined, X_combined.shape[1])
        X_sequences, y_sequences = split_sequences(X_flat, y_flat, window_size, step)
    else:
        # Data is already in correct window size
        X_sequences = X_combined
        y_sequences = y_combined.astype(int)
    
    # Normalize data
    n_samples, n_timestamps, n_features = X_sequences.shape
    X_reshaped = X_sequences.reshape((n_samples * n_timestamps, n_features))
    scaler = StandardScaler()
    X_reshaped = scaler.fit_transform(X_reshaped)
    X_sequences = X_reshaped.reshape((n_samples, n_timestamps, n_features))
    
    # One-hot encode labels
    num_classes = 6
    y_onehot = tf.keras.utils.to_categorical(y_sequences, num_classes=num_classes)
    
    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                      'SITTING', 'STANDING', 'LAYING']
    
    print(f"UCI HAR - X_sequences shape: {X_sequences.shape}")
    print(f"UCI HAR - y_sequences shape: {y_sequences.shape}")
    
    return X_sequences, y_sequences, y_onehot, activity_labels

def load_wisdm_data(data_path, window_size=80, step=40):
    """
    Load WISDM dataset
    
    Args:
        data_path (str): Path to WISDM dataset file
        window_size (int): Window size for sequences
        step (int): Step size for overlapping windows
        
    Returns:
        tuple: (X_sequences, y_sequences, y_onehot, activity_labels)
    """
    print("Loading WISDM dataset...")
    
    # Find the correct data file
    if os.path.isdir(data_path):
        possible_files = ['WISDM_ar_v1.1_raw.txt', 'WISDM_ar_v1.1.txt']
        for file in possible_files:
            file_path = os.path.join(data_path, file)
            if os.path.exists(file_path):
                data_path = file_path
                break
    
    # Parse the file line by line
    column_names = ['user', 'activity', 'timestamp', 'x_accel', 'y_accel', 'z_accel']
    rows = []
    
    with open(data_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.endswith(';'):
                line = line[:-1]
            if line:
                try:
                    values = line.split(',')
                    if len(values) == 6:
                        rows.append(values)
                except Exception as e:
                    continue
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=column_names)
    
    # Convert numeric columns
    for col in ['user', 'timestamp', 'x_accel', 'y_accel', 'z_accel']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Encode activity labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['activity'])
    
    # Extract features (accelerometer data)
    features = df[['x_accel', 'y_accel', 'z_accel']].values
    
    # Create sequences
    X_sequences, y_sequences = split_sequences(features, encoded_labels, window_size, step)
    
    # Normalize features
    n_samples, n_timestamps, n_features = X_sequences.shape
    X_reshaped = X_sequences.reshape((n_samples * n_timestamps, n_features))
    scaler = StandardScaler()
    X_reshaped = scaler.fit_transform(X_reshaped)
    X_sequences = X_reshaped.reshape((n_samples, n_timestamps, n_features))
    
    # One-hot encode labels
    num_classes = len(label_encoder.classes_)
    y_onehot = tf.keras.utils.to_categorical(y_sequences, num_classes=num_classes)
    
    # Create activity labels list
    activity_labels = label_encoder.classes_.tolist()
    
    print(f"WISDM - X_sequences shape: {X_sequences.shape}")
    print(f"WISDM - y_sequences shape: {y_sequences.shape}")
    print(f"WISDM - Activities: {activity_labels}")
    
    return X_sequences, y_sequences, y_onehot, activity_labels

def load_mhealth_data(data_path, window_size=50, step=25):
    """
    Load mHealth dataset
    
    Args:
        data_path (str): Path to mHealth dataset folder
        window_size (int): Window size for sequences
        step (int): Step size for overlapping windows
        
    Returns:
        tuple: (X_sequences, y_sequences, y_onehot, activity_labels)
    """
    print("Loading mHealth dataset...")
    
    # Find all log files in the folder
    log_files = glob.glob(os.path.join(data_path, "*.log"))
    
    if not log_files:
        raise FileNotFoundError(f"No log files found in {data_path}")
    
    print(f"Found {len(log_files)} log files")
    
    # Process each file and combine data
    all_data = []
    
    for log_file in log_files:
        print(f"Processing {os.path.basename(log_file)}")
        data = pd.read_csv(log_file, header=None, sep='\t')
        all_data.append(data)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")
    
    # Extract features and labels
    # The last column (index 23) is the activity label
    X = combined_data.iloc[:, :-1].values
    y_original = combined_data.iloc[:, -1].values
    
    # Filter out records with activity label 0 (no activity)
    mask = y_original != 0
    X = X[mask]
    y_original = y_original[mask]
    
    # Shift class labels down by 1 to start from 0
    y = y_original - 1
    
    # Create sequences
    X_sequences, y_sequences = split_sequences(X, y, window_size, step)
    
    # Normalize data
    n_samples, n_timestamps, n_features = X_sequences.shape
    X_reshaped = X_sequences.reshape((n_samples * n_timestamps, n_features))
    scaler = StandardScaler()
    X_reshaped = scaler.fit_transform(X_reshaped)
    X_sequences = X_reshaped.reshape((n_samples, n_timestamps, n_features))
    
    # Get unique activity labels
    unique_labels = np.unique(y_sequences)
    num_classes = len(unique_labels)
    
    # One-hot encode labels
    y_onehot = tf.keras.utils.to_categorical(y_sequences, num_classes=num_classes)
    
    activity_labels = [
        "Standing still", "Sitting and relaxing", "Lying down", "Walking",
        "Climbing stairs", "Waist bends forward", "Frontal elevation of arms",
        "Knees bending (crouching)", "Cycling", "Jogging", "Running", "Jump front & back"
    ]
    
    print(f"mHealth - X_sequences shape: {X_sequences.shape}")
    print(f"mHealth - y_sequences shape: {y_sequences.shape}")
    print(f"mHealth - Number of classes: {num_classes}")
    
    return X_sequences, y_sequences, y_onehot, activity_labels

def load_pamap2_data(data_path, window_size=100, step=50):
    """
    Load PAMAP2 dataset
    
    Args:
        data_path (str): Path to PAMAP2 dataset
        window_size (int): Window size for sequences
        step (int): Step size for overlapping windows
        
    Returns:
        tuple: (X_sequences, y_sequences, y_onehot, activity_labels)
    """
    print("Loading PAMAP2 dataset...")
    
    # Define column names
    columns = [
        "timestamp", "activityID", "heart_rate",
        "hand_temp", "hand_acc_x", "hand_acc_y", "hand_acc_z",
        "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
        "hand_mag_x", "hand_mag_y", "hand_mag_z",
        "chest_temp", "chest_acc_x", "chest_acc_y", "chest_acc_z",
        "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
        "chest_mag_x", "chest_mag_y", "chest_mag_z",
        "ankle_temp", "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
        "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
        "ankle_mag_x", "ankle_mag_y", "ankle_mag_z"
    ]
    
    # Find Protocol data files
    protocol_path = os.path.join(data_path, "Protocol")
    protocol_files = [f for f in os.listdir(protocol_path) if f.endswith(".dat")]
    
    # Find Optional data files
    optional_path = os.path.join(data_path, "Optional")
    optional_files = []
    if os.path.exists(optional_path):
        optional_files = [f for f in os.listdir(optional_path) if f.endswith(".dat")]
    
    all_data = []
    
    # Process Protocol files
    for file in protocol_files:
        subject_id = int(file.split("subject")[1].split(".dat")[0])
        print(f"Processing (Protocol): {file}")
        file_path = os.path.join(protocol_path, file)
        with open(file_path, "r") as f:
            for line in f:
                row = [x for x in line.strip().split(" ") if x != ""]
                if len(row) >= len(columns):
                    try:
                        data_row = [float(x) for x in row[:len(columns)]]
                        data_row.append(subject_id)
                        all_data.append(data_row)
                    except ValueError:
                        continue
    
    # Process Optional files
    for file in optional_files:
        subject_id = int(file.split("subject")[1].split(".dat")[0])
        print(f"Processing (Optional): {file}")
        file_path = os.path.join(optional_path, file)
        with open(file_path, "r") as f:
            for line in f:
                row = [x for x in line.strip().split(" ") if x != ""]
                if len(row) >= len(columns):
                    try:
                        data_row = [float(x) for x in row[:len(columns)]]
                        data_row.append(subject_id)
                        all_data.append(data_row)
                    except ValueError:
                        continue
    
    # Convert to DataFrame
    columns.append("subjectID")
    df = pd.DataFrame(all_data, columns=columns)
    
    # Activity mapping
    activity_mapping = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
        12: 7, 13: 8, 16: 9, 17: 10, 24: 11
    }
    
    # Filter and map activities
    df = df[df["activityID"].isin(activity_mapping.keys())]
    df["activityID"] = df["activityID"].map(activity_mapping)
    
    # Drop timestamp and temperature columns
    df.drop(["timestamp", "hand_temp", "heart_rate", "chest_temp", "ankle_temp"], 
            axis=1, inplace=True)
    
    # Handle missing values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Extract features and labels
    X = df.drop(["activityID", "subjectID"], axis=1).values
    y = df["activityID"].values
    
    # Create sequences
    X_sequences, y_sequences = split_sequences(X, y, window_size, step)
    
    # Normalize data
    n_samples, n_timestamps, n_features = X_sequences.shape
    X_reshaped = X_sequences.reshape((n_samples * n_timestamps, n_features))
    scaler = StandardScaler()
    X_reshaped = scaler.fit_transform(X_reshaped)
    X_sequences = X_reshaped.reshape((n_samples, n_timestamps, n_features))
    
    # Get unique activity labels
    unique_labels = np.unique(y_sequences)
    num_classes = len(unique_labels)
    
    # One-hot encode labels
    y_onehot = tf.keras.utils.to_categorical(y_sequences, num_classes=num_classes)
    
    activity_labels = [
        "lying", "sitting", "standing", "walking", "running", "cycling",
        "Nordic walking", "ascending stairs", "descending stairs", 
        "vacuum cleaning", "ironing", "rope jumping"
    ]
    
    print(f"PAMAP2 - X_sequences shape: {X_sequences.shape}")
    print(f"PAMAP2 - y_sequences shape: {y_sequences.shape}")
    print(f"PAMAP2 - Number of classes: {num_classes}")
    
    return X_sequences, y_sequences, y_onehot, activity_labels

def load_dataset(dataset_name, data_path):
    """
    Load dataset based on dataset name
    
    Args:
        dataset_name (str): Name of dataset ('UCI_HAR', 'WISDM', 'mHealth', 'PAMAP2')
        data_path (str): Path to dataset
        
    Returns:
        tuple: (X_sequences, y_sequences, y_onehot, activity_labels)
    """
    dataset_config = get_dataset_config(dataset_name)
    window_size = dataset_config['window_size']
    step = dataset_config['step']
    
    if dataset_name == 'UCI_HAR':
        return load_uci_har_data(data_path, window_size, step)
    elif dataset_name == 'WISDM':
        return load_wisdm_data(data_path, window_size, step)
    elif dataset_name == 'mHealth':
        return load_mhealth_data(data_path, window_size, step)
    elif dataset_name == 'PAMAP2':
        return load_pamap2_data(data_path, window_size, step)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")