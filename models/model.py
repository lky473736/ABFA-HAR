#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from config import *

class StemLayer(tf.keras.layers.Layer):
    def __init__(self, filters_list, **kwargs):
        super(StemLayer, self).__init__(**kwargs)
        self.filters_list = filters_list

    def build(self, input_shape):
        self.conv_layers = []
        self.bn_layers = []
        self.relu_layers = []
        
        for i, filters in enumerate(self.filters_list):
            self.conv_layers.append(
                layers.Conv1D(
                    filters, 
                    kernel_size=1, 
                    activation=None,
                    name=f'stem_conv_{i+1}'
                )
            )
            self.bn_layers.append(
                layers.BatchNormalization(
                    momentum=BN_MOMENTUM, 
                    name=f'stem_bn_{i+1}'
                )
            )
            self.relu_layers.append(
                layers.ReLU(name=f'stem_relu_{i+1}')
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for conv, bn, relu in zip(self.conv_layers, self.bn_layers, self.relu_layers):
            x = conv(x)
            x = bn(x, training=training)
            x = relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'filters_list': self.filters_list})
        return config

class MKTC(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_sizes, **kwargs):
        super(MKTC, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes

    def build(self, input_shape):
        self.branch_convs = []
        self.branch_bns = []
        
        branch_filters = self.filters // len(self.kernel_sizes)
        
        for i, k in enumerate(self.kernel_sizes):
            self.branch_convs.append(
                layers.Conv1D(
                    filters=branch_filters,
                    kernel_size=k,
                    padding='same',
                    activation='relu',
                    name=f'MKTC_conv_{k}'
                )
            )
            self.branch_bns.append(
                layers.BatchNormalization(
                    momentum=BN_MOMENTUM,
                    name=f'MKTC_bn_{k}'
                )
            )
        
        self.concat = layers.Concatenate(name='MKTC_concat')
        self.proj_conv = layers.Conv1D(self.filters, kernel_size=1, name='MKTC_proj')
        self.final_bn = layers.BatchNormalization(momentum=BN_MOMENTUM, name='MKTC_final_bn')
        self.final_relu = layers.ReLU(name='MKTC_final_relu')
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        branches = []
        for conv, bn in zip(self.branch_convs, self.branch_bns):
            branch = conv(inputs)
            branch = bn(branch, training=training)
            branches.append(branch)
        
        x = self.concat(branches)
        x = self.proj_conv(x)
        x = self.final_bn(x, training=training)
        x = self.final_relu(x)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes
        })
        return config

class ABFA(tf.keras.layers.Layer):
    def __init__(self, filters, activity_classes, dropout_rate=ABFA_DROPOUT_RATE, **kwargs):
        super(ABFA, self).__init__(**kwargs)
        self.filters = filters
        self.activity_classes = activity_classes
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.action_prototypes = self.add_weight(
            name='action_prototypes',
            shape=(self.activity_classes, input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.embedding_proj = layers.Dense(input_shape[-1], name='embedding_proj')
        self.augment_proj = layers.Dense(input_shape[-1], name='augment_proj')
        
        self.bn = layers.BatchNormalization(momentum=BN_MOMENTUM, name='abfa_bn')
        self.dropout = layers.Dropout(self.dropout_rate, name='abfa_dropout')
        self.layer_norm = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name='abfa_layer_norm')
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        x_proj = self.embedding_proj(inputs)
        
        proto_similarity = tf.einsum('btd,cd->btc', x_proj, self.action_prototypes)
        proto_attention = tf.nn.softmax(proto_similarity, axis=-1)
        
        enhanced = tf.einsum('btc,cd->btd', proto_attention, self.action_prototypes)
        augmented = self.augment_proj(enhanced)
        
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
    def __init__(self, filters, kernel_sizes=MSA_KERNEL_SIZES, **kwargs):
        super(MSA_Block, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes

        if len(kernel_sizes) == 3:
            self.filter_dims = [filters // 4, filters // 4, filters // 2]
        else:
            self.filter_dims = [filters // len(kernel_sizes)] * len(kernel_sizes)

    def build(self, input_shape):
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
        
        x = layers.Concatenate(name='MSA_concat')(outputs)
        
        return self.output_proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes
        })
        return config

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ffn_units, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ffn_units = ffn_units

    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.key_dim,
            name='transformer_attention'
        )
        
        self.add_1 = layers.Add(name='transformer_add_1')
        self.norm_1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name='transformer_norm_1')
        
        self.ffn_1 = layers.Dense(self.ffn_units, activation='relu', name='transformer_ffn_1')
        self.ffn_2 = layers.Dense(self.ffn_units, name='transformer_ffn_2')
        
        self.add_2 = layers.Add(name='transformer_add_2')
        self.norm_2 = layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name='transformer_norm_2')
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs, inputs, training=training)
        
        x = self.add_1([inputs, attn_output])
        x = self.norm_1(x, training=training)
        
        ffn_output = self.ffn_1(x)
        ffn_output = self.ffn_2(ffn_output)
        
        x = self.add_2([x, ffn_output])
        x = self.norm_2(x, training=training)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'ffn_units': self.ffn_units
        })
        return config

class ClassificationHead(tf.keras.layers.Layer):
    def __init__(self, units_list, num_classes, dropout_rate, **kwargs):
        super(ClassificationHead, self).__init__(**kwargs)
        self.units_list = units_list
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.global_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')
        
        self.dense_layers = []
        for i, units in enumerate(self.units_list):
            self.dense_layers.append(
                layers.Dense(units, activation='relu', name=f'cls_dense_{i+1}')
            )
        
        self.dropout = layers.Dropout(self.dropout_rate, name='cls_dropout')
        self.predictions = layers.Dense(self.num_classes, activation='softmax', name='predictions')
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.global_pool(inputs)
        
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            if i == 0:
                x = self.dropout(x, training=training)
        
        return self.predictions(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units_list': self.units_list,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        })
        return config

def build_abfa_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    x = StemLayer(
        filters_list=INITIAL_PROJECTION_FILTERS,
        name='stem_layer'
    )(inputs)
    
    x = MKTC(
        filters=MKTC_FILTERS,
        kernel_sizes=MKTC_KERNEL_SIZES,
        name='mktc_block'
    )(x)
    
    x = ABFA(
        filters=ABFA_FILTERS, 
        activity_classes=num_classes,
        name='abfa_block'
    )(x)
    
    x = MSA_Block(
        filters=MSA_FILTERS, 
        kernel_sizes=MSA_KERNEL_SIZES,
        name='MSA_block'
    )(x)
    
    x = TransformerEncoder(
        num_heads=TRANSFORMER_HEADS,
        key_dim=TRANSFORMER_KEY_DIM,
        ffn_units=TRANSFORMER_UNITS,
        name='transformer_encoder'
    )(x)
    
    outputs = ClassificationHead(
        units_list=CLASSIFICATION_UNITS,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        name='classification_head'
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ABFA_Full_Model')
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model_summary(model):
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_name': model.name
    }