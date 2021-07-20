import ignite.metrics as metrics
from ignite.contrib.metrics import ROC_AUC
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Union, List
import numpy as np
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
import utils
import metrics as mymetrics
from scipy import stats


# make visible to other namespace
BCELoss = torch.nn.BCELoss
MSELoss = torch.nn.MSELoss
MAELoss = torch.nn.L1Loss

class Loss(metrics.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        if output == None:
            return
        super().update(output)

class Precision(metrics.Precision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        if output == None:
            return
        super().update(output)


class Recall(metrics.Precision):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        if output == None:
            return
        super().update(output)

class MeanAveragePrecision(ROC_AUC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        if output == None:
            return
        super().update(output)

def _d_prime(auc):
    return stats.norm().ppf(auc) * np.sqrt(2.0)

class Dprime(ROC_AUC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        if output == None:
            return
        super().update(output)

    def compute(self):
        auc = super().compute()
        return _d_prime(auc)

class Intersection_F1(metrics.Metric):
    """docstring for Intersection_F1"""
    def __init__(self,
                 encoder,
                 validation_data_file: Path,
                 validation_duration_file: Path,
                 output_transform=lambda x: x,
                 n_thresholds: int = 5):
        self.validation_data_file = validation_data_file
        self.validation_duration_file = validation_duration_file
        self.test_thresholds = np.arange(
            1 / (n_thresholds * 2), 1, 1 /
            n_thresholds)
        self.encoder = encoder
        self.filenames = []
        self.y_pred = []
        super(Intersection_F1, self).__init__(output_transform=output_transform)

    def reset(self):
        self.filenames = []
        self.y_pred = []

    def update(self, output):
        if output == None:
            return
        y_pred, filename = output
        y_pred = y_pred.detach().cpu()
        self.filenames.extend(filename)
        for i in range(y_pred.shape[0]):
            self.y_pred.append(y_pred[i])

    def compute(self):
        thresholded_predictions = utils.frame_preds_to_chunk_preds(self.y_pred,
                                               self.filenames,
                                               self.encoder,
                                               thresholds=self.test_thresholds,
                                               frame_resolution=0.01
                                               )
        intersection_f1_macro = mymetrics.compute_per_intersection_macro_f1(
            thresholded_predictions, self.validation_data_file,
            self.validation_duration_file)
        return intersection_f1_macro




if __name__ == "__main__":
    pass
