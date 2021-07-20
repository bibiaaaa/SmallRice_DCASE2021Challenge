import torch
import random
import numpy as np
import pandas as pd
from typing import Dict, List
from h5py import File
from torch.multiprocessing import Queue
from scipy import ndimage
import utils


class WeakHDF5Dataset(torch.utils.data.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(
        self,
        data_frame: pd.DataFrame,
        label_encoder,
    ):
        super(WeakHDF5Dataset, self).__init__()
        self._dataframe = data_frame
        self._datasetcache = {}
        self._len = len(self._dataframe)
        self._label_encoder = label_encoder

        filename, hdf5path = self._dataframe.iloc[0][['filename', 'hdf5path']]
        with File(hdf5path, 'r') as store:
            self.datadim = store[filename].shape[-1]

    def __len__(self):
        return self._len

    def __del__(self):
        for k, cache in self._datasetcache.items():
            cache.close()

    def __getitem__(self, index: int):
        fname, label_names, hdf5path = self._dataframe.iloc[index][[
            'filename', 'event_labels', 'hdf5path'
        ]]
        #Generate target from int list [1,5,7] --> [0,1,0,0,0,1,0,1]
        target = self._label_encoder.encode(label_names)
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')

        data = self._datasetcache[hdf5path][f"{fname}"][:].astype(np.float32)
        return torch.as_tensor(data, dtype=torch.float32), target, fname

class UnlabeledHDF5Dataset(WeakHDF5Dataset):

    def __init__(self, data_frame):
        super(UnlabeledHDF5Dataset, self).__init__(data_frame,
                                                   label_encoder=None)

    def __getitem__(self, index:int):
        fname, hdf5path = self._dataframe.iloc[index][[
            'filename', 'hdf5path'
        ]]
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')

        data = self._datasetcache[hdf5path][f"{fname}"][:].astype(np.float32)
        return torch.as_tensor(data, dtype=torch.float32), fname

class StrongHDF5Dataset(WeakHDF5Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(
            self,
            data_frame: pd.DataFrame,
            label_encoder,
            smooth_ramp:int = 0,
            default_duration:float = 10.,
            **feature_args,
    ):
        super(StrongHDF5Dataset, self).__init__(data_frame, label_encoder)
        self.hop_size = feature_args.get('hop_size', 160)
        self.sr = feature_args.get('sr', 16000)
        self.win_size = feature_args.get('win_size', 512)
        self.default_duration = default_duration * self.sr
        self.smooth_ramp = smooth_ramp
        assert smooth_ramp == 0 or smooth_ramp %2 ==1, "Smoothing can only be done with 3,5,7 kernelsizes .."
        if self.smooth_ramp>0:
            self.smooth_kernel = np.ones((self.smooth_ramp, 1), dtype=np.float32) / self.smooth_ramp

    def __getitem__(self, index: int):
        fname, onsets, offsets, label_names, hdf5path = self._dataframe.iloc[index][[
            'filename', 'onset', 'offset', 'event_label', 'hdf5path'
        ]]


        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')

        data = self._datasetcache[hdf5path][f"{fname}"][:].astype(np.float32)
        #Generate target from int list [1,5,7] --> [0,1,0,0,0,1,0,1]
        #When center = False
        # n_frames = 1 + int((data.shape[-1] - self.win_size) / self.hop_size)
        # When center = True this applies
        # n_frames = data.shape[-1] // self.hop_size + 1
        n_frames = int(self.default_duration // self.hop_size + 1)
        frame_duration = data.shape[-1]/ self.sr / n_frames
        strong_target = np.zeros((n_frames,
                                  len(self._label_encoder)))
        for on, off, label in zip(onsets, offsets, label_names):
            idx = self._label_encoder.get_index_for_label(label)
            on, off = int(on // frame_duration), int(off//frame_duration)
            # Smoothes the targets, such that there is a target "ramp" before and after a frame target. For example, the output [0,0,1,1,0,0] will be transformed to  [0.33,0.66,1,1,1,0.66,0.33] 
            if self.smooth_ramp > 0:
                pad = self.smooth_ramp - 1 
                strong_target[max(on - pad,0): min(off+pad, n_frames), idx] = 1
                strong_target = ndimage.convolve(strong_target, self.smooth_kernel, mode='constant').clip(0, 1)
            else:
                # print(on, off, idx)
                strong_target[on:off, idx] = 1.

        weak_target = self._label_encoder.encode(set(label_names))
        return torch.as_tensor(data, dtype=torch.float32), torch.as_tensor(
            strong_target, dtype=torch.float32), torch.as_tensor(weak_target, dtype=torch.float32), fname


class MultiDataLoader(torch.utils.data.IterableDataset):
    
    def __init__(self, datasets: Dict[str,
                                      torch.utils.data.DataLoader]):
        self.dataloaders = datasets
        self.dataloader_iters = {k: iter(v) for k,v in self.dataloaders.items()}

    def __iter__(self):
        while True:
            datas = {}
            for key in self.dataloader_iters:
                try:
                    batch = next(self.dataloader_iters[key])
                except StopIteration:
                    # Reset iterator
                    self.dataloader_iters[key] = iter(self.dataloaders[key])
                    batch = next(self.dataloader_iters[key])
                datas[key] = batch
            yield datas
            datas = []

def pad(tensorlist:List[torch.Tensor], padding_value:float=0.):
    # Tensors are expected to be B, ..., T
    lengths = [f.shape[-1] for f in tensorlist]
    dims = tensorlist[0].shape
    trailing_dims = dims[:-1]
    batch_dim = len(lengths)
    num_raw_samples = max(lengths)
    out_dims = (batch_dim,) + trailing_dims +  (num_raw_samples,)
    out_tensor = torch.full(out_dims,
                         fill_value=padding_value,
                         dtype=torch.float32)
    for i, tensor in enumerate(tensorlist):
        length = tensor.shape[-1]
        out_tensor[i,..., :length] = tensor[...,:length]
    return out_tensor

class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, labels: List[List[str]], encoder , random_state= None):
        self._random_state = np.random.RandomState(seed= random_state)
        unique_labels = np.unique(np.concatenate(labels))
        n_labels = len(unique_labels)
        label_to_idx_list = [[] for _ in range(n_labels)]
        label_to_length = []
        for idx, lbs in enumerate(labels):
            for lb in lbs:
                lb = encoder.get_index_for_label(lb)
                label_to_idx_list[lb].append(idx)
        for i in range(len(label_to_idx_list)):
            label_to_idx_list[i] = np.array(label_to_idx_list[i])
            self._random_state.shuffle(label_to_idx_list[i])
            label_to_length.append(len(label_to_idx_list[i]))
        self.label_to_idx_list = label_to_idx_list
        self.label_to_length = label_to_length
        self._num_classes = len(encoder)
        self.pointers_of_classes = [0] * self._num_classes
        self._len = len(labels)
        self.queue = Queue()

    def getitemindex(self, lab_idx: int):
        '''
        returns next index, given a label index
        '''
        cur_item = self.pointers_of_classes[lab_idx]
        self.pointers_of_classes[lab_idx] += 1
        index = self.label_to_idx_list[lab_idx][cur_item]
        #Reshuffle and reset points if overlength
        if self.pointers_of_classes[lab_idx] >= self.label_to_length[
                lab_idx]:  #Reset
            self.pointers_of_classes[lab_idx] = 0
            self._random_state.shuffle(self.label_to_idx_list[lab_idx])
        return index

    def populate_queue(self):
        # Can be overwritten by subclasses
        classes_set = np.arange(self._num_classes).tolist()
        self._random_state.shuffle(classes_set)
        for c in classes_set:
            self.queue.put(c)  # Push to queue class indices


    def __iter__(self):
        while True:
            if self.queue.empty():
                self.populate_queue()
            lab_idx = self.queue.get()  # Get next item, single class index
            index = self.getitemindex(lab_idx)
            yield index

    def __len__(self):
        return self._len

def sequential_collate(batches:List):
    data, *targets, fnames = zip(*batches)
    targets = tuple(map(lambda x: torch.stack(x), targets))
    return pad(data), *targets, fnames

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def getdataloader(dataset,  **kwargs):
    return torch.utils.data.DataLoader(dataset,
                                collate_fn=sequential_collate,
                                worker_init_fn=seed_worker,
                                **kwargs
                                )


def getmultidatasetdataloader(**dataloaders):
    return MultiDataLoader(datasets=dataloaders)


if __name__ == "__main__":
    import utils
    import models
    syn_df = utils.read_labels('data/synthetic_train.tsv')
    weak_df = utils.read_labels('data/weak_train.tsv')
    encoder = utils.LabelEncoder(syn_df['event_label'].sum())
    default_feature_params = {
        'n_mels': 64,
        'hop_size': 160,
        'win_size': 512,
        'f_min': 0
    }
    front_end = models.MelSpectrogramFrontEnd(**default_feature_params)
    # for b, wt, f in getdataloader(WeakHDF5Dataset(weak_df, encoder), sampler=BalancedSampler(weak_df['event_labels'], encoder)):
        # x = front_end(b)
        # pass

    for b, st,wt, f in getdataloader(StrongHDF5Dataset(syn_df, encoder), sampler=BalancedSampler(syn_df['event_label'], encoder)):
        x = front_end(b)
        pass
