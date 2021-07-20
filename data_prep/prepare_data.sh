LABEL_OUTPUT_ROOT=$PWD/"labels"
HDF_OUTPUT_ROOT=${PWD}/"hdf5"


function dump_data(){
    echo "Dumping data from $1 to $2"
    python3 dump_raw_to_hdf.py $1 -o $2
}

function process_Task2(){
    ROOT_DIR=$(readlink -e "Task2/data/")

    output_hdf="${HDF_OUTPUT_ROOT}/Task2/eval_source_test.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task2/eval_source_test.tsv"
    find ${ROOT_DIR}/eval/*/source_test -type f -name "*wav"  | python3 -c "import sys; f=sys.stdin.readlines(); print('filename\tmachinetype\tsection\tdomain'); print('\n'.join([a.rstrip('\n') + '\t' + a.split('/')[10] + '\t' + a.split('/')[-1][8:10] + '\t' + a.split('/')[-2].split('_')[0] for a in f]))" > ${output_labels}
    dump_data ${output_labels} ${output_hdf}
    tmpfile=$(mktemp)
    cat ${output_labels} | awk -vH=${output_hdf} 'NR==1{print $0"\thdf5path"}NR>1{print $0"\t"H}'  > $tmpfile
    mv $tmpfile ${output_labels}

    output_hdf="${HDF_OUTPUT_ROOT}/Task2/eval_target_test.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task2/eval_target_test.tsv"
    find ${ROOT_DIR}/eval/*/target_test -type f -name "*wav"  | python3 -c "import sys; f=sys.stdin.readlines(); print('filename\tmachinetype\tsection\tdomain'); print('\n'.join([a.rstrip('\n') + '\t' + a.split('/')[10] + '\t' + a.split('/')[-1][8:10] + '\t' + a.split('/')[-2].split('_')[0] for a in f]))" > ${output_labels}
    dump_data ${output_labels} ${output_hdf}
    tmpfile=$(mktemp)
    cat ${output_labels} | awk -vH=${output_hdf} 'NR==1{print $0"\thdf5path"}NR>1{print $0"\t"H}'  > $tmpfile
    mv $tmpfile ${output_labels}


    output_hdf="${HDF_OUTPUT_ROOT}/Task2/dev_train.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task2/dev_train.tsv"
    find ${ROOT_DIR}/dev/*/train -type f -name "*wav"  | python3 -c "import sys; f=sys.stdin.readlines(); print('filename\tmachinetype\tsection\tdomain\tlabel'); print('\n'.join([a.rstrip('\n') + '\t' + a.split('/')[10] + '\t' + a.split('_')[1] + '\t' + a.split('_')[2] + '\t' +  a.split('_')[4]  for a in f]))" > ${output_labels}
    dump_data ${output_labels} ${output_hdf}

    output_hdf="${HDF_OUTPUT_ROOT}/Task2/dev_source.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task2/dev_source.tsv"
    find ${ROOT_DIR}/dev/*/source_test -type f -name "*wav"  | python3 -c "import sys; f=sys.stdin.readlines(); print('filename\tmachinetype\tsection\tdomain\tlabel'); print('\n'.join([a.rstrip('\n') + '\t' + a.split('/')[10] + '\t' + a.split('_')[2] + '\t' + a.split('_')[3] + '\t' +  a.split('_')[5]  for a in f]))" > ${output_labels}
    # Append HDF5PATH
    tmpfile=$(mktemp)
    cat ${output_labels} | awk -vH=${output_hdf} 'NR==1{print $0"\thdf5path"}NR>1{print $0"\t"H}'  > $tmpfile
    mv $tmpfile ${output_labels}

    dump_data ${output_labels} ${output_hdf}

    output_hdf="${HDF_OUTPUT_ROOT}/Task2/dev_target.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task2/dev_target.tsv"
    find ${ROOT_DIR}/dev/*/target_test -type f -name "*wav" | python3 -c "import sys; f=sys.stdin.readlines(); print('filename\tmachinetype\tsection\tdomain\tlabel'); print('\n'.join([a.rstrip('\n') + '\t' + a.split('/')[10] + '\t' + a.split('_')[2] + '\t' + a.split('_')[3] + '\t' +  a.split('_')[5]  for a in f]))" > ${output_labels}
    # Append HDF5PATH
    tmpfile=$(mktemp)
    cat ${output_labels} | awk -vH=${output_hdf} 'NR==1{print $0"\thdf5path"}NR>1{print $0"\t"H}'  > $tmpfile
    mv $tmpfile ${output_labels} 

    dump_data ${output_labels} ${output_hdf}

    output_hdf="${HDF_OUTPUT_ROOT}/Task2/additional_eval_train.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task2/additional_eval_train.tsv"
    find ${ROOT_DIR}/additional_data/*/train -type f -name "*wav"  | python3 -c "import sys; f=sys.stdin.readlines(); print('filename\tmachinetype\tsection\tdomain\tlabel'); print('\n'.join([a.rstrip('\n') + '\t' + a.split('/')[10] + '\t' + a.split('_')[1] + '\t' + a.split('_')[2] + '\t' +  a.split('_')[4]  for a in f]))" > ${output_labels}
    # Append HDF5PATH
    tmpfile=$(mktemp)
    cat ${output_labels} | awk -vH=${output_hdf} 'NR==1{print $0"\thdf5path"}NR>1{print $0"\t"H}'  > $tmpfile
    mv $tmpfile ${output_labels}

    dump_data ${output_labels} ${output_hdf}
}

function process_Task4_SED(){
    ROOT_DIR=$(readlink -e "Task4/data/")

    output_hdf="${HDF_OUTPUT_ROOT}/Task4/validation.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task4/validation.tsv"
    output_duration="${LABEL_OUTPUT_ROOT}/Task4/validation_duration.tsv"
    cat ${ROOT_DIR}/raw_datasets/desed_real/metadata/validation/validation.tsv | awk -vH=${output_hdf} -vR=${ROOT_DIR}/raw_datasets/desed_real/audio/validation/ 'NR==1{print $0"\thdf5path"}NR>1{print R$0"\t"H}' > ${output_labels}
    function get_duration(){
        echo -e "$1\t$(soxi -D $1)"
    }
    export -f get_duration
    awk 'NR>1{print $1}' $output_labels | sort -u | parallel get_duration | awk 'BEGIN{print "filename\tduration"}{print $1"\t"$2}' > ${output_duration}


    dump_data ${output_labels} ${output_hdf}

    output_hdf="${HDF_OUTPUT_ROOT}/Task4/weak_train.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task4/weak_train.tsv"
    cat ${ROOT_DIR}/raw_datasets/desed_real/metadata/train/weak.tsv | awk -vH=${output_hdf} -vR=${ROOT_DIR}/raw_datasets/desed_real/audio/train/weak/ 'NR==1{print $0"\thdf5path"}NR>1{print R$0"\t"H}'  > ${output_labels}
    dump_data ${output_labels} ${output_hdf}

    output_hdf="${HDF_OUTPUT_ROOT}/Task4/unlabaled_train.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task4/unlabeled_train.tsv"
    cat ${ROOT_DIR}/raw_datasets/desed_real/metadata/train/unlabel_in_domain.tsv | awk -vH=${output_hdf} -vR=${ROOT_DIR}/raw_datasets/desed_real/audio/train/unlabeled_in_domain/ 'NR==1{print $0"\thdf5path"}NR>1{print R$0"\t"H}'  > ${output_labels}
    dump_data ${output_labels} ${output_hdf}


    #Synthetic data
    output_hdf="${HDF_OUTPUT_ROOT}/Task4/synthetic_train.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task4/synthetic_train.tsv"
    cat ${ROOT_DIR}/dcase2021/dataset/metadata/train/synthetic21_train/soundscapes.tsv | awk -vH=${output_hdf} -vR=${ROOT_DIR}/dcase2021/dataset/audio/train/synthetic21_train/soundscapes/ 'NR==1{print $0"\thdf5path"}NR>1{print R$0"\t"H}'  > ${output_labels}
    dump_data ${output_labels} ${output_hdf}

    output_hdf="${HDF_OUTPUT_ROOT}/Task4/synthetic21_validation.h5"
    output_labels="${LABEL_OUTPUT_ROOT}/Task4/synthetic_validation.tsv"
    cat ${ROOT_DIR}/dcase2021/dataset/metadata/validation/synthetic21_validation/soundscapes.tsv | awk -vH=${output_hdf} -vR=${ROOT_DIR}/dcase2021/dataset/audio/validation/synthetic21_validation/soundscapes/ 'NR==1{print $0"\thdf5path"}NR>1{print R$0"\t"H}'  > ${output_labels}
    dump_data ${output_labels} ${output_hdf}
}

mkdir -p ${LABEL_OUTPUT_ROOT}/Task{2,4}
mkdir -p ${HDF_OUTPUT_ROOT}/Task{2,4}

if [[ $# != 1 ]]; then
    echo "Input [task2, task4]"
    exit
fi

if [[ $1 == "task2" ]]; then
    process_Task2
    
fi
if [[ $1 == "task4" ]]; then
    process_Task4_SED
fi
echo "Unknown input parameter"


