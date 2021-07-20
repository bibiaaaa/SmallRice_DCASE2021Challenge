import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from einops import rearrange
from pathlib import Path
from loguru import logger
from collections import OrderedDict
import torch_audiomentations as wavtransforms
from torch import Tensor
from typing import Union, List, Dict
import scipy
import torchaudio.transforms as audio_transforms
import sys


def getlogger(outputfile: str = None):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def parse_wavtransforms(transforms_dict: Dict):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    return torch.nn.Sequential(*[
        getattr(wavtransforms, trans_name)(**v)
        for trans_name, v in transforms_dict.items()
    ])


def parse_spectransforms(transforms_dict: Dict):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    return torch.nn.Sequential(*[
        getattr(audio_transforms, trans_name)(**v)
        for trans_name, v in transforms_dict.items()
    ])


def read_data_with_segments(path: str, segments: str):
    data_df = pd.read_csv(path, sep='\t').dropna().convert_dtypes(
    )  #drops some indices during evaluation which have no labels
    segment_df = pd.read_csv(segments, sep='\t').dropna().convert_dtypes()
    df = data_df.merge(segment_df, on='filename')
    return df


def read_labels(path: str, aggregate: bool = True):
    df = pd.read_csv(path, sep='\t').dropna(
    )  #drops some indices during evaluation which have no labels
    #Weag data
    if 'event_labels' in df:
        df['event_labels'] = df['event_labels'].str.split(',')
    #Stronk data
    if ('event_label' in df.columns) and ('onset' in df.columns) and (
            'offset' in df.columns) and ('hdf5path' in df.columns) and aggregate:
        # Aggregate to filename -> [[on1, on2],[off1, off2], [l1,l2]]
        df = df.groupby('filename').agg({
            'onset': list,
            'offset': list,
            'event_label': list,
            'hdf5path': lambda x: x.iloc[0],
        }).reset_index()
    #Elif only for predicted labels without hdf5path
    elif ('event_label' in df.columns) and ('onset' in df.columns) and (
            'offset' in df.columns) and aggregate:
        df = df.groupby('filename').agg({
            'onset': list,
            'offset': list,
            'event_label': list,
        }).reset_index()
    return df


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    """
    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]
    # Shift change_index with one, focus on frame after the change.
    change_indices += 1
    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]
    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]
    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))



class LabelEncoder(object):

    def __init__(self, labels: List = None):
        super(LabelEncoder, self).__init__()
        if labels is not None:
            self.fit(labels)

    def fit(self, labels: List):
        self.val_to_index = {
            val: index
            for index, val in enumerate(np.unique(labels))
        }
        self.index_to_val = {
            index: val
            for val, index in self.val_to_index.items()
        }
        self.num_classes_ = len(self.val_to_index)
        return self

    def get_index_for_label(self, label: str) -> int:
        return self.val_to_index[label]

    def encode_from_id(self, lab_id: int, device=None):
        idx_tensor = torch.tensor(self.val_to_index[self.index_to_val[lab_id]],
                                  device=device).clone().detach()
        return torch.zeros(self.num_classes_,
                           device=device).scatter(0, idx_tensor, 1)

    def encode(self, labels: str, device=None) -> Tensor:
        idxs = torch.tensor([self.val_to_index[v] for v in labels])
        return torch.zeros(self.num_classes_,
                           device=device).scatter_(0, idxs, 1)

    def decode(self, vector: Tensor) -> List:
        assert vector.ndim == 1, "Only 1d data supported"
        return [self.index_to_val[idx.item()] for idx in torch.nonzero(vector).flatten()]

    def decode_strong(self, x):
        result = []
        for i, label_col in enumerate(x.T):
            change_indices = find_contiguous_regions(label_col)
            for start, end in change_indices:
                result.append((self.index_to_val[i], start, end))
        return result

    def __len__(self):
        return self.num_classes_

    def __repr__(self):
        return f"Labelencoder with {self.num_classes_} classes :{list(self.val_to_index.keys())}"

    def state_dict(self):
        return self

    def load_state_dict(self, state):
        return self



class _DictWrapper(object):
    def __init__(self, adict):
        self.dict = adict
    def state_dict(self):
        return self.dict
    def load_state_dict(self, state):
        self.dict = state

# Obtained on the validation (dev) set. Just the average frame-length
# for each repsective class
base_adaptive_filter_sizes = {
    'Alarm_bell_ringing': 195,
    'Blender': 522,
    'Cat': 138,
    'Dishes': 62,
    'Dog': 140,
    'Electric_shaver_toothbrush': 773,
    'Frying': 825,
    'Running_water': 524,
    'Speech': 149,
    'Vacuum_cleaner': 848
}


def adaptive_median_filter(x: List[torch.Tensor], encoder, factor: float = 3):
    preds = []
    for pred in x:
        for ind, name in encoder.index_to_val.items():
            pred[..., ind] = torch.as_tensor(
                scipy.ndimage.filters.median_filter(
                    pred.cpu().numpy()[..., ind],
                    base_adaptive_filter_sizes[name] // factor))
        preds.append(pred)
    return preds


def median_filter(x: List[torch.Tensor], filter_size: int):
    preds = []
    for pred in x:
        pred = torch.as_tensor(
            scipy.ndimage.filters.median_filter(pred.cpu().numpy(),
                                                (filter_size, 1)))
        preds.append(pred)
    return preds


def frame_preds_to_chunk_preds(
    frame_preds: List[torch.Tensor],
    filenames: List,
    encoder,
    thresholds: List[float] = None,
    frame_resolution: float = 0.01,
):
    """ Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary
    Args:
        strong_preds: torch.Tensor, batch of strong predictions.
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int, the number of frames for which to apply median window (smoothing).
        pad_indx: list, the list of indexes which have been used for padding.
    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    # Init a dataframe per threshold
    assert len(filenames) == len(frame_preds)
    prediction_dfs = {}
    if thresholds is None:
        thresholds = [0.5]
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for i in tqdm(range(len(frame_preds))):  # over batches
        for c_th in thresholds:
            pred = frame_preds[i].detach().cpu().numpy()
            pred = pred > c_th
            #Post processing via median filtering
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred,
                                columns=["event_label", "onset", "offset"])
            pred['onset'], pred['offset'] = pred[
                'onset'] * frame_resolution, pred['offset'] * frame_resolution
            pred["filename"] = filenames[i]
            prediction_dfs[c_th] = prediction_dfs[c_th].append(
                pred, ignore_index=True)
    return prediction_dfs


def split_train_cv(df:pd.DataFrame, frac:float):
    train_df = df.sample(frac=frac, random_state=42)
    cv_df = df[~df.index.isin(train_df.index)].reset_index(drop=True)
    return train_df.reset_index(drop=True), cv_df


def mixup(x: torch.Tensor, lamb: torch.Tensor):
    if x.shape[0] % 2 != 0:
        return x
    x_transposed = rearrange(x, 'b ... d -> d ... b')
    return rearrange(
        (x_transposed[..., 0::2] * lamb) + (x_transposed[..., 1::2] *
                                            (1. - lamb)), 'd ... b -> b ... d')

def load_pretrained(model, trained_model):
    model_dict = model.state_dict()
    # filter unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in trained_model.items() if (k in model_dict) and (
            model_dict[k].shape == trained_model[k].shape)
    }
    assert len(pretrained_dict) > 0, "Loading failed!"
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return model


if __name__ == "__main__":
    import torch
    enc = LabelEncoder(["a",'B','c','d'])
    torch.save(enc.state_dict(),'/tmp/a')
    f = torch.load('/tmp/a')
    enc.load_state_dict(f)
    # x = enc.encode(['a','B'])
    # print(enc.decode(x))

    # x = torch.randn(4, 5)
    # lamb  = torch.randn(2)
    # y = mixup(x, lamb)
    # print(y.shape)
