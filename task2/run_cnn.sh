#!/bin/bash
machine_types="fan"
config_file=mobilenetv2
datenow=`date '+%Y%m%d_%H%M%S'`
basepath=experiments/${config_file}/${datenow}
for machine_type in $machine_types;
do
    outputpath=${basepath}/${machine_type}
    mkdir -p ${outputpath}
    echo ${outputpath} 
    CUDA_VISIBLE_DEVICES=0 python run_simple.py --config_file=run_config/${config_file}.yaml --datenow=\"${datenow}\" --basepath=${basepath} --max_epochs=100 --num_workers=16 train_supervised ${machine_type} 2>&1 
done
