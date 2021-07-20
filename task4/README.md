# Task 4 Experiments

Before running the experiments, all data needs to be firstly processed using the scripts in `data_prep/`.

Then just link the `labels` dir into the current directory:

```bash
ln -s ../data_prep/labels/Task4/ data

```

The experiments are all controlled by the script `run.py`

To run our baseline experiments please execute:

```bash
python3 run.py run configs/train_supervised_baseline.yaml # Runs with weak + synthetic only
python3 run.py run configs/train_baseline.yaml # Runs the above + UDA
python3 run.py run configs/train_baseline_specaug.yaml # Runs the above + UDA + Specaug
```





