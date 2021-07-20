#!/bin/bash
machine_types="fan gearbox slider ToyTrain ToyCar pump valve"
config_file=ae
datenow=`date '+%Y%m%d_%H%M%S'`
basepath=experiments/${config_file}/${datenow}
i=0
for machine_type in $machine_types;
do
    outputpath=${basepath}/${machine_type}
    mkdir -p ${outputpath}
    echo ${outputpath} 
    CUDA_VISIBLE_DEVICES=${i} nohup python run_simple.py --config_file=run_config/${config_file}.yaml --datenow=\"${datenow}\" --basepath=${basepath} --max_epochs=300 --num_workers=10 --seed=616 train_single_model ${machine_type} > experiments/${config_file}/${datenow}/${machine_type}/run.log 2>&1 & 
    i=$(($i+1))
done
