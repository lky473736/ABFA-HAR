#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from config import *

class ABFA(tf.keras.layers.Layer):
    """
    Component 3: ABFA Block
    """
    def __init__(self, filters, activity_classes, dropout_rate=ABFA_DROPOUT_RATE, **kwargs):
        super(ABFA, self).__init__(**kwargs)
        self.filters = filters
        self.activity_classes = activity_classes
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Action prototypes for each activity class
        self.action_prototypes = self.add_weight(
            name='action_prototypes',
            shape=(self.activity_classes, input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Projection layers
        self.embedding_proj = layers.Dense(input_shape[-1], name='embedding_proj')
        self.augment_proj = layers.Dense(input_shape[-1], name='augment_proj')
        
        # Normalization and regularization
        self.bn = layers.BatchNormalization(momentum=BN_MOMENTUM, name='abfa_bn')
        self.dropout = layers.Dropout(self.dropout_rate, name='abfa_dropout')
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name='abfa_layer_norm')
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Project input embeddings
        x_proj = self.embedding_proj(inputs)
        
        # Compute similarity with action prototypes
        proto_similarity = tf.einsum('btd,cd->btc', x_proj, self.action_prototypes)
        proto_attention = tf.nn.softmax(proto_similarity, axis=-1)
        
        # Prototype-guided feature enhancement
        enhanced = tf.einsum('btc,cd->btd', proto_attention, self.action_prototypes)
        augmented = self.augment_proj(enhanced)
        
        # Residual connection with normalization
        x = self.bn(inputs + augmented, training=training)
        x = self.dropout(x, training=training)
        
        return self.layer_norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'activity_classes': self.activity_classes,
            'dropout_rate': self.dropout_rate
        })
        return config

class MSA_Block(layers.Layer):
    """
    Component 4: MSA Block
    """
    def __init__(self, filters, kernel_sizes=MSA_KERNEL_SIZES, **kwargs):
        super(MSA_Block, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes

        # Distribute filters across different scales
        if len(kernel_sizes) == 3:
            self.filter_dims = [filters // 4, filters // 4, filters // 2]
        else:
            self.filter_dims = [filters // len(kernel_sizes)] * len(kernel_sizes)

    def build(self, input_shape):
        # Depthwise convolutions for different temporal scales
        self.conv_layers = []
        for i, k_size in enumerate(self.kernel_sizes):
            self.conv_layers.append(
                layers.DepthwiseConv1D(
                    kernel_size=k_size,
                    strides=1,
                    padding='same',
                    depth_multiplier=1,
                    name=f'MSA_depthwise_conv_{k_size}'
                )
            )
        
        # Projection layers for each scale
        self.proj_layers = []
        for i, filter_dim in enumerate(self.filter_dims):
            self.proj_layers.append(
                layers.Conv1D(
                    filter_dim, 
                    kernel_size=1, 
                    padding='same',
                    name=f'MSA_proj_{i}'
                )
            )
        
        # Output projection
        self.output_proj = layers.Conv1D(
            self.filters, 
            kernel_size=1,
            name='MSA_output_proj'
        )
        
        super().build(input_shape)

    def call(self, inputs):
        outputs = []
        for conv, proj in zip(self.conv_layers, self.proj_layers):
            x = conv(inputs)
            x = proj(x)
            outputs.append(x)
        
        # Concatenate multi-scale features
        x = layers.Concatenate(name='MSA_concat')(outputs)
        
        # Final projection
        return self.output_proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes
        })
        return config

def build_abfa_model(input_shape, num_classes):
    """
    Build the complete ABFA model with all components (1+2+3+4+5+6)
    
    Args:
        input_shape (tuple): Input shape (time_steps, features)
        num_classes (int): Number of activity classes
        
    Returns:
        tf.keras.Model: Complete ABFA model
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = inputs

    # Component 1: Initial Projection
    x = layers.Conv1D(
        INITIAL_PROJECTION_FILTERS[0], 
        kernel_size=1, 
        activation=None,
        name='initial_proj_1'
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name='initial_bn_1')(x)
    x = layers.ReLU(name='initial_relu_1')(x)
    
    x = layers.Conv1D(
        INITIAL_PROJECTION_FILTERS[1], 
        kernel_size=1, 
        activation=None,
        name='initial_proj_2'
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name='initial_bn_2')(x)
    x = layers.ReLU(name='initial_relu_2')(x)

    # Component 2: MKTC
    multi_scale_outputs = []
    for i, k in enumerate(MKTC_KERNEL_SIZES):
        branch = layers.Conv1D(
            filters=MKTC_FILTERS // len(MKTC_KERNEL_SIZES),
            kernel_size=k,
            padding='same',
            activation='relu',
            name=f'MKTC_conv_{k}'
        )(x)
        branch = layers.BatchNormalization(
            momentum=BN_MOMENTUM,
            name=f'MKTC_bn_{k}'
        )(branch)
        multi_scale_outputs.append(branch)

    x = layers.Concatenate(name='MKTC_concat')(multi_scale_outputs)
    x = layers.Conv1D(MKTC_FILTERS, kernel_size=1, name='MKTC_proj')(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name='MKTC_final_bn')(x)
    x = layers.ReLU(name='MKTC_final_relu')(x)

    # Component 3: ABFA Block
    x = ABFA(
        filters=ABFA_FILTERS, 
        activity_classes=num_classes,
        name='abfa_block'
    )(x)

    # Component 4: MSA Block
    x = MSA_Block(
        filters=MSA_FILTERS, 
        kernel_sizes=MSA_KERNEL_SIZES,
        name='MSA_block'
    )(x)

    # Component 5: Transformer Encoder Block
    # Multi-head attention
    attn_output = layers.MultiHeadAttention(
        num_heads=TRANSFORMER_HEADS, 
        key_dim=TRANSFORMER_KEY_DIM,
        name='transformer_attention'
    )(x, x)
    
    # Add & Norm
    x = layers.Add(name='transformer_add_1')([x, attn_output])
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name='transformer_norm_1')(x)
    
    # Feed Forward Network
    ffn_output = layers.Dense(TRANSFORMER_UNITS, activation='relu', name='transformer_ffn_1')(x)
    ffn_output = layers.Dense(TRANSFORMER_UNITS, name='transformer_ffn_2')(ffn_output)
    
    # Add & Norm
    x = layers.Add(name='transformer_add_2')([x, ffn_output])
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name='transformer_norm_2')(x)

    # Component 6: Classification Head
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x = layers.Dense(CLASSIFICATION_UNITS[0], activation='relu', name='cls_dense_1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='cls_dropout')(x)
    x = layers.Dense(CLASSIFICATION_UNITS[1], activation='relu', name='cls_dense_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='ABFA_Full_Model')
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile the ABFA model with optimizer and loss function
    
    Args:
        model (tf.keras.Model): ABFA model to compile
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        tf.keras.Model: Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model_summary(model):
    """
    Get detailed model summary including parameter counts
    
    Args:
        model (tf.keras.Model): Model to summarize
        
    Returns:
        dict: Model summary information
    """
    # Calculate parameter counts
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_name': model.name
    }