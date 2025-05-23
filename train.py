#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split

from config import *
from models.model import build_abfa_model, compile_model
from data_parsing.data_loader import load_dataset

def train_model(dataset_name, data_path):
    """
    Train ABFA model on specified dataset
    
    Args:
        dataset_name (str): Name of dataset ('UCI_HAR', 'WISDM', 'mHealth', 'PAMAP2')
        data_path (str): Path to dataset
    """
    # Set random seed for reproducibility
    set_seed(SEED)
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    num_classes = dataset_config['num_classes']
    window_size = dataset_config['window_size']
    
    print(f"Training ABFA model on {dataset_name} dataset")
    print(f"Number of classes: {num_classes}")
    print(f"Window size: {window_size}")
    
    # Load dataset
    X_sequences, y_sequences, y_onehot, activity_labels = load_dataset(
        dataset_name, data_path
    )
    
    print(f"Data shape: {X_sequences.shape}")
    print(f"Labels shape: {y_onehot.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_onehot, test_size=0.2, random_state=SEED, stratify=y_sequences
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_abfa_model(input_shape, num_classes)
    model = compile_model(model, LEARNING_RATE)
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    )
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save model
    model_name = f"{dataset_name.lower()}_model.h5"
    model.save(model_name)
    print(f"Model saved as {model_name}")
    
    return model, history, (X_test, y_test)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python train.py <dataset_name>")
        print("Dataset names: UCI_HAR, WISDM, mHealth, PAMAP2")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    data_path = ''
    if dataset_name == 'UCI_HAR' : 
        data_path = 'data/UCI_HAR_Dataset'
    if dataset_name == 'WISDM' : 
        data_path = 'data/WISDM_ar_v1.1'
    if dataset_name == 'PAMAP2' : 
        data_path = 'data/PAMAP2_Dataset'
    if dataset_name == 'mHealth' : 
        data_path = 'data/MHEALTHDATASET'
    
    if dataset_name not in ['UCI_HAR', 'WISDM', 'mHealth', 'PAMAP2']:
        print(f"Invalid dataset name: {dataset_name}")
        print("Valid options: UCI_HAR, WISDM, mHealth, PAMAP2")
        sys.exit(1)
    
    # Train model
    model, history, test_data = train_model(dataset_name, data_path)