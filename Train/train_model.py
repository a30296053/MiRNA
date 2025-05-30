##################################################################################################################
import os
import pickle
import datetime
import numpy as np
import pandas as pd
##################################################################################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-gpu", "--gpu", type=int, default=0, help="gpu usage")
parser.add_argument('-batch', '--batch', type=int, default=64, help='batch-size')
parser.add_argument("-lr", "--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument('-fol', '--fol', type=int, default=0, help='fol')
parser.add_argument('-epochs', '--epochs', type=int, default=300, help='epochs')
parser.add_argument('-model', '--model', type=int, default=0, help='model')
parser.add_argument('-dt', '--dt', type=int, default=0, choices=[0, 1], help='model')
parser.add_argument('-n', '--n', type=int, default=1, help='n_inf')
args = parser.parse_args()
# python train_0415.py -gpu=0 -batch=64 -lr=1e-3 -dt=0 -fol=3 -epochs=300 -model=6 -n=1
# python train_0415.py -gpu=0 -batch=64 -lr=1e-3 -dt=1 -fol=0 -epochs=300 -model=6 -n=1
##################################################################################################################

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    try:
        tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

##################################################################################################################
from models import *

lr = args.lr
batch_size = args.batch
dt = args.dt
fol = args.fol
epochs = args.epochs
n_inf = args.n

if dt == 0:
    data_dt = 'miRaw' 
    dout=0.2
elif dt == 1:
    data_dt = 'DeepMirTar'
    dout=0.4
else:
    print('incorrect dataset')
    assert False

data_dict = 'data_full_0411.pickle'
##################################################################################################################
def get_Seq_train(data, data_dt, fol):

    idxs = np.arange(4).tolist()
    idxs.remove(fol)
    
    Seq_train = []
    Seq_train_label = []
    
    for i in idxs:
        x, y = data[data_dt][i]
        Seq_train.append(x)
        Seq_train_label.append(y)
        
    Seq_train = np.concatenate(Seq_train, axis=0)
    Seq_train_label = np.concatenate(Seq_train_label, axis=0)
    return Seq_train, Seq_train_label

def load_data(data_dict, data_dt, fol):

    with open(data_dict, 'rb') as handle:
        data= pickle.load(handle)
    
    assert data_dt in data
    assert fol < 4
    Seq_ind, Seq_ind_label = data[data_dt][4]
    
    Seq_ind_label = np.float32(Seq_ind_label)
    print(Seq_ind.shape, Seq_ind_label.shape, Seq_ind_label.dtype, Seq_ind_label[:2])
    
    Seq_val, Seq_val_label = data[data_dt][fol]
    Seq_val_label = np.float32(Seq_val_label)
    print(Seq_val.shape, Seq_val_label.shape, Seq_val_label.dtype, Seq_val_label[:2])
    
    Seq_train, Seq_train_label = get_Seq_train(data, data_dt, fol)
    Seq_train_label = np.float32(Seq_train_label)
    print(Seq_train.shape, Seq_train_label.shape, Seq_train_label.dtype, Seq_train_label[:2])

    return Seq_train, Seq_train_label, Seq_val, Seq_val_label, Seq_ind, Seq_ind_label
##################################################################################################################

X_train, y_train, X_valid, y_valid, X_ind, y_ind = load_data(data_dict, data_dt, fol)

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).cache()
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).cache()
ds_val = ds_val.batch(batch_size)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

ds_ind = tf.data.Dataset.from_tensor_slices((X_ind, y_ind)).cache()
ds_ind = ds_ind.batch(batch_size)
ds_ind = ds_ind.prefetch(tf.data.AUTOTUNE)

##################################################################################################################

for _ in range(n_inf):
    
    if args.model == 0:
        model = org_model()
        model_name = 'miraw_org'
    elif args.model == 1:
        model = model_v1(dout=dout)
        model_name = 'model_v1'
    elif args.model == 2:
        model = model_v2(dout=dout)
        model_name = 'model_v2'
    elif args.model == 3:
        model = model_v3(dout=dout)
        model_name = 'model_v3'
    elif args.model == 4:
        model = model_v4(dout=dout)
        model_name = 'model_v4'
    elif args.model == 5:
        model = model_v5()
        model_name = 'model_v5'
    elif args.model == 6:
        model = model_v6()
        model_name = 'model_v6'
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr), 
        metrics=['accuracy',
                 tf.keras.metrics.F1Score(threshold=0.5, average='macro')]
    )
    model.summary()
    
    ##################################################################################################################
    data_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_model_path = os.path.join('Trained_model', data_dt, model_name + '_FOL' + str(fol), data_time_str )
    os.makedirs(save_model_path,exist_ok=True)
    
    model_checkpoints_acc = os.path.join(save_model_path, model_name + '_best_acc.h5')
    checkpoint_best_callback_acc = tf.keras.callbacks.ModelCheckpoint(model_checkpoints_acc, monitor='val_accuracy', verbose=1, 
                                                             save_best_only=True, save_weights_only=False, 
                                                             mode='max')
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_accuracy', factor=0.1, patience=20, verbose=1, mode='max', min_lr=1e-8)
    stop_callback = tf.keras.callbacks.EarlyStopping( monitor='val_accuracy', patience=50, verbose=1, mode='max', restore_best_weights=True)
    callbacks=[checkpoint_best_callback_acc, lr_callback, stop_callback]
    
    ##################################################################################################################
    
    
    history = model.fit(ds_train, 
                        epochs=epochs, 
                        validation_data=ds_val, 
                        callbacks=callbacks,
                        verbose=2)
    
    ##################################################################################################################
    
    his_json_name = os.path.join(save_model_path, model_name + '.json')
    hist_df = pd.DataFrame(history.history) 
    with open(his_json_name, mode='w') as f:
        hist_df.to_json(f)
    
    print('evaluate indepentdent testing data: ')
    model.evaluate(ds_ind)

