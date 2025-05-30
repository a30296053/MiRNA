import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def data_extract(DeepMirTar_txt, dt='DeepMirTar'):

    encode = dict(zip('NAUCG', range(5)))

    df = pd.read_csv(DeepMirTar_txt, sep="\t", header=None)
    #print(df.iloc[0])
    assert dt in ['DeepMirTar', 'miRaw']
    if dt=='DeepMirTar':
        df = df.drop(df.index[0]).reset_index(drop=True)

    df = df[df[3].str.len()<=40] ## remove 52_samples miRNA are more than 40

    max_RNA = max(df[1].str.len().max(), 26)
    max_miRNA = max(df[3].str.len().max(), 40)
    print(max_RNA, max_miRNA)
    
    df[1] = [x + 'N'*(max_RNA-len(x)) for x in df[1].tolist()]
    df[3] = [x.replace('T','U') + 'N'*(max_miRNA-len(x)) for x in df[3].tolist()]
    df[5] = df[1] + df[3]
    df['data'] = df[5].apply(lambda x: np.array([encode[a.upper()] for a in x], np.int64))
    
    DeepMirTar_data = np.stack(df['data'])
    DeepMirTar_label = np.stack(df[4])
    DeepMirTar_label = np.expand_dims(DeepMirTar_label, -1)
    DeepMirTar_label = np.int64(DeepMirTar_label)
    return DeepMirTar_data, DeepMirTar_label

def get_Seq_train(data, dt, fol):

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
# ensure the data can be Evenly distribute to the different fold
skf = StratifiedKFold(n_splits=5, shuffle=True)
data = {}
for dt in ['DeepMirTar', 'miRaw']:
    data_txt = 'data/data_miRaw_noL_noMisMissing_remained_seed1122.txt' if dt=='miRaw' else 'data/data_DeepMirTar_removeMisMissing_remained_seed1122.txt'
    data_txt_ind = 'data/data_miRaw_noL_noMisMissing_indTest_seed1122_Unique.txt' if dt=='miRaw' else 'data/data_DeepMirTar_test.txt'
    
    data[dt] = {}
    Seq_data, Seq_label = data_extract(data_txt, dt=dt)
    for i, (_, fol_index) in enumerate(skf.split(Seq_data, Seq_label)):
        data[dt][i] = [Seq_data[fol_index], Seq_label[fol_index]]

    Seq_data_ind, Seq_label_ind = data_extract(data_txt_ind, dt=dt)
    data[dt][5] = [Seq_data_ind, Seq_label_ind]

    with open('data_full_0411.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
