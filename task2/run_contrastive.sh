#!/bin/bash
machine_type="fan"
config_file=contrastive
datenow=`date '+%Y%m%d_%H%M%S'`
basepath=experiments/${config_file}/${datenow}
outputpath=${basepath}/${machine_type}
mkdir -p ${outputpath}
echo ${outputpath} 
CUDA_VISIBLE_DEVICES=0 python run_cnn_contrastive.py --config_file=run_config/${config_file}.yaml --datenow=\"${datenow}\" --basepath=${basepath} --max_epochs=100 --num_workers=15 train_cl ${machine_type} 2>&1
