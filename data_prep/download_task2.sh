#!/usr/bin/env bash
standard_data_links=(
    https://zenodo.org/record/4562016/files/dev_data_fan.zip?download=1
    https://zenodo.org/record/4562016/files/dev_data_gearbox.zip?download=1
    https://zenodo.org/record/4562016/files/dev_data_pump.zip?download=1
    https://zenodo.org/record/4562016/files/dev_data_slider.zip?download=1
    https://zenodo.org/record/4562016/files/dev_data_ToyCar.zip?download=1
    https://zenodo.org/record/4562016/files/dev_data_ToyTrain.zip?download=1
    https://zenodo.org/record/4562016/files/dev_data_valve.zip?download=1
    )

additional_data_links=(
    https://zenodo.org/record/4660992/files/eval_data_fan_train.zip?download=1
    https://zenodo.org/record/4660992/files/eval_data_gearbox_train.zip?download=1
    https://zenodo.org/record/4660992/files/eval_data_pump_train.zip?download=1
    https://zenodo.org/record/4660992/files/eval_data_slider_train.zip?download=1
    https://zenodo.org/record/4660992/files/eval_data_ToyCar_train.zip?download=1
    https://zenodo.org/record/4660992/files/eval_data_ToyTrain_train.zip?download=1
    https://zenodo.org/record/4660992/files/eval_data_valve_train.zip?download=1
    )

eval_data_links=(
    https://zenodo.org/record/4884786/files/eval_data_fan_test.zip?download=1
    https://zenodo.org/record/4884786/files/eval_data_gearbox_test.zip?download=1
    https://zenodo.org/record/4884786/files/eval_data_pump_test.zip?download=1
    https://zenodo.org/record/4884786/files/eval_data_slider_test.zip?download=1
    https://zenodo.org/record/4884786/files/eval_data_ToyCar_test.zip?download=1
    https://zenodo.org/record/4884786/files/eval_data_ToyTrain_test.zip?download=1
    https://zenodo.org/record/4884786/files/eval_data_valve_test.zip?download=1
)


function download() {
    echo "Starting to download $1"

    output_file=$(echo $1 | awk -F['?/'] '{print $(NF-1)}')
    # Unzip if sucessful
    if wget -q $1 --continue -O ${2}/${output_file}; then
        unzip ${2}/${output_file} -d ${2}
    fi
}

export -f download
# Output dirs
mkdir -p Task2/data/{dev,eval,additional_data}

echo "Downloading development data"
for i in "${standard_data_links[@]}";
do
    download $i Task2/data/dev/
done

echo "Downloading additional data"
for i in "${additional_data_links[@]}";
do
    download $i Task2/data/additional_data/
done

echo "Downloading eval data"
for i in "${eval_data_links[@]}";
do
    download $i Task2/data/eval
done

