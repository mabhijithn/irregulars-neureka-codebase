"""Train a U-net for a specific data view.

This file loads a pre-processed data object, creates a new U-net and starts the training loop on the data.
During training, the network weights with the best validation performance are saved to disk.
"""
# Importing of necessary libraries
import h5py
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from utils import SegmentGenerator, setup_tf, build_unet


# Variables locating the pre-processed data and the to-be-saved network weights
save_path = 'PATH_TO_LOADED_DATA.h5'
network_path = 'PATH_TO_NETWORK_WEIGHTS.h5'

# Some random variables
fs = 200
n_secs = 30
n_channels = 18

# Tensorflow function to detect GPU properly
setup_tf()

# Loading data + normalizing
with h5py.File(save_path, 'r') as f:
    file_names = []
    labels = []
    signals = []
    
    file_names_ds = f['filenames']
    signals_ds = f['signals']
    labels_ds = f['labels']
    
    for i in range(len(signals_ds)):
        file_names.append(file_names_ds[i])
        data = np.asarray(np.vstack(signals_ds[i]), dtype=np.float32).T
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        signals.append((data-mean)/std)
        labels.append(labels_ds[i])

# Building a list for stratified train-val split, balance the amount of files containing seizures
seizure_label = []
for label in labels:
    seizure_label.append(np.sum(label)>0)
seizure_label = np.asarray(seizure_label, dtype=np.uint8)

# Train-val split
signal_train, signal_val, label_train, label_val = model_selection.train_test_split(signals, labels,
                                 test_size=0.2, random_state=1337, stratify=seizure_label)

# Network settings
n_filters = 8
window_size = 4096
n_channels = 18

# Build the unet, and unet_train objects (use same underlying layers, unet_train is used for deep supervision)
unet, unet_train = build_unet(window_size=window_size, n_channels=n_channels, n_filters=n_filters)

# Build a Keras data generator object
generator = SegmentGenerator(signals=signal_train, labels=label_train, batch_size=32,
                             window_size=window_size, stride=1000, n_channels=n_channels)

# Training settings + bookkeeping variables
n_batches = len(generator)
n_epochs = 100
optimizer = Adam(lr=1e-4)

loss_train = np.zeros(shape=(n_epochs,))
xentr_train = np.zeros(shape=(n_epochs,))
xentr_val_mean = np.zeros(shape=(n_epochs,))
xentr_val_std = np.zeros(shape=(n_epochs,))
one = np.float32(1.)
bin_xent = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)

best_loss = 1e20

# Loss weights
all_labels = np.copy(np.concatenate(label_train))
n_bckg = np.sum(all_labels==0)
n_seiz = np.sum(all_labels==1)
del all_labels


# Actual training loop
for epoch in range(n_epochs):
    
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_xentr_avg = tf.keras.metrics.Mean()
    print('====== Epoch #{0:3d} ======'.format(epoch))
    # Training loop over all batches
    for batch in range(n_batches):
        x, y = generator.__getitem__(batch)
        
        with tf.GradientTape() as t:
            y_0, y_1, y_2, y_3, y_4, y_5 = unet_train(x, training=True)
            xentr0 = bin_xent(y_true=tf.expand_dims(y, axis=-1), y_pred=tf.expand_dims(y_0, axis=-1),
                             sample_weight=n_bckg/n_seiz*y+(one-y))
            y = y[:, ::4]
            xentr1 = bin_xent(y_true=tf.expand_dims(y, axis=-1), y_pred=tf.expand_dims(y_1, axis=-1),
                             sample_weight=n_bckg/n_seiz*y+(one-y))
            y = y[:, ::4]
            xentr2 = bin_xent(y_true=tf.expand_dims(y, axis=-1), y_pred=tf.expand_dims(y_2, axis=-1),
                             sample_weight=n_bckg/n_seiz*y+(one-y))
            y = y[:, ::4]
            xentr3 = bin_xent(y_true=tf.expand_dims(y, axis=-1), y_pred=tf.expand_dims(y_3, axis=-1),
                             sample_weight=n_bckg/n_seiz*y+(one-y))
            y = y[:, ::4]
            xentr4 = bin_xent(y_true=tf.expand_dims(y, axis=-1), y_pred=tf.expand_dims(y_4, axis=-1),
                             sample_weight=n_bckg/n_seiz*y+(one-y))
            y = y[:, ::4]
            xentr5 = bin_xent(y_true=tf.expand_dims(y, axis=-1), y_pred=tf.expand_dims(y_5, axis=-1),
                             sample_weight=n_bckg/n_seiz*y+(one-y))

            loss0 = tf.reduce_mean(xentr0)
            loss1 = tf.reduce_mean(xentr1)
            loss2 = tf.reduce_mean(xentr2)
            loss3 = tf.reduce_mean(xentr3)
            loss4 = tf.reduce_mean(xentr4)
            loss5 = tf.reduce_mean(xentr5)
            loss = loss0 + 0.2*(loss1 + loss2 + loss3 + loss4 + loss5)
            
        grad = t.gradient(loss, unet_train.trainable_variables)
        optimizer.apply_gradients(zip(grad, unet_train.trainable_variables))
        epoch_loss_avg(loss)
        epoch_xentr_avg(loss0)
    
    generator.on_epoch_end()
    
    xentr_train[epoch] = epoch_xentr_avg.result()
    loss_train[epoch] = epoch_loss_avg.result()
    print('Loss Train     - {0:.4f}'.format(loss_train[epoch]))
    print('Xentropy Train - {0:.4f}'.format(xentr_train[epoch]))
    
    # Calculating validation loss for "early stopping"
    xentr = []
    for j in range(len(signal_val)):
        signal = signal_val[j]
        label = label_val[j]
        if len(signal) < window_size:
            break
        x = []
        y = []
        for i in range(len(signal)//window_size):
            x.append(signal[window_size*i:(i+1)*window_size, :])
            y.append(label[window_size*i:(i+1)*window_size])
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        y_ = unet.predict(x)
        xentr.append(bin_xent(y_true=tf.expand_dims(y, axis=-1), y_pred=tf.expand_dims(y_, axis=-1),
                             sample_weight=n_bckg/n_seiz*y+(one-y)).numpy())
        
    xentr_val_mean[epoch] = np.mean(xentr)
    xentr_val_std[epoch] = np.std(xentr)
    
    if xentr_val_mean[epoch] < best_loss:
        best_loss = xentr_val_mean[epoch]
        unet.save_weights(network_path)
    
    print('Xentropy Val   - {0:.4f} ± {1:.4f}'.format(xentr_val_mean[epoch], xentr_val_std[epoch]))