# miRNA-Target-Identification.
We developed an ATTbiLSTM-MMP, including Conv1D blocks, attention block, and multi-kernel max-pooling layers, to improve the accuracy of identifying important biomarkers.
Our datasets are obtained from two datasets in two studies, DeepMirTar (https://academic.oup.com/bioinformatics/article/34/22/3781/5026656) and miRAW (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006185), and refer to a study miTAR (https://doi.org/10.1186/s12859-021-04026-6)

## Experimental setup
In training, we utilized the standard binary cross-entropy loss function, specifically tailored for binary classification. We employed the AdamW [28] optimization algorithm with an initial learning rate of 1e-3 and a batch size of 64. If there wasn’t any significant improvement since the last 20th epoch, the learning rate dropped by a factor of 10. We trained the model for 300 epochs with early stopping to avoid overfitting; the patience for early stopping was set to 50 epochs.

## Train the model
In the file "Train", there is processing_data, which can be used to package data into pickle files for easy reading during training. In this study, the data is packaged into pickle files, for research models.py contains the models adjusted during the research process.
The process of training the model can be seen through train_model.py. If there are related parameters that need to be adjusted, they are adjusted from these two codes. The following is an example of the command for training the model:
```python=
python train_0415.py -gpu=0 -batch=64 -lr=1e-3 -dt=0 -fol=0 -epochs=300 -model=4 -n=10
```
The description of each parameter can be found in train_model.py
The parameters of the training process and the weights of the model are stored in the "Trained_model" folder. Since the file is too large, only one copy of the training results of ATTbiLSTM-MMP in each data set is stored here.

## Evaluation Result
In the file "Train", "process_result" can be used to compare the results of all data between models, while "evaluation" simply outputs the average comparison of multiple training results after training.

## Requirements
numpy
pandas
sklearn
tqdm
tensorflow
keras
argparse

## Report issues
If you have any questions or suggestions or encounter any issues, please contact a30296053@gmail.com
