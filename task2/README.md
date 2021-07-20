# Jointly Training Framework for DCASE 2021 Challenge Task 2 

Before running the experiments, all data needs to be firstly processed using the scripts in `data_prep/`.

Then link the labels and hdf5 files into the current dictionary:
```bash
ln -s ../data_prep/labels/Task2/ data/labels
ln -s ../data_prep/data/Task2/ data/hdf5
```

The experiments are all controlled by the script `train.py`

To run our baseline experiments please execute the bash scripts: `run_ae.sh`,`run_cnn.sh` and `run_contrastive.sh` 
 
To modify the training configs, please refer to yaml files in `configs/`.