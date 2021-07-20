# Download and prepare data


First install the requirements to store the data:

```bash
python3 -m pip install -r requirements.txt
```


## Task 2

First download the data:

```bash
bash download_task2.sh
```

which will dump the data into `Task2/data/`

And then in order to prepare the data for training, we dump all data into HDF5 files:

```bash
bash prepare_data.sh task2
```


## Task4

Downloading the data uses the script provided by the committee:

```bash
mkdir -p Task4/data/

wget https://raw.githubusercontent.com/DCASE-REPO/DESED_task/master/recipes/dcase2021_task4_baseline/generate_dcase_task4_2021.py -P Task4/data/
cd Task4/data/ && python3 generate_dcase_task4_2021.py
```


After downloading, run:

```bash
bash prepare_data.sh task4
```
