import sys
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
import librosa


class HDF5_trainset(tdata.Dataset):
    def __init__(
        self,
        hdf5_path,
        data_frame,
        machine_type,
        domain,
        section=[0, 1, 2, 3, 4, 5],
        acoustic_paras={
            'n_frames': 5,
            'hop_frames': 1,
            'n_mels': 128,
            'hop_size': 512,
            'n_fft': 1024
        },
    ) -> None:
        super(HDF5_trainset, self).__init__()
        self._machinetype = machine_type
        self.section = section
        self.domain = domain
        self._dataframe = data_frame[
            (data_frame.machinetype == self._machinetype)
            & (data_frame.section.isin(self.section)) &
            (data_frame.domain.isin(self.domain))].reset_index()
        self.n_frames = acoustic_paras.get('n_frames', 5)
        self.n_mels = acoustic_paras.get('n_mels', 128)
        self.hop_size = acoustic_paras.get('hop_size', 512)
        self.n_fft = acoustic_paras.get('n_fft', 1024)
        self.hop_frames = acoustic_paras.get('hop_frames', 1)
        self.dims = self.n_frames * self.n_mels
        self.datacache = None
        self.h5path = hdf5_path
        self.chunk_length = self.n_fft + (self.n_frames - 1) * self.hop_size
        self.index_to_file = []
        self.wav_frames = []
        self.index_to_chunkindex = []
        for index, fname in enumerate(self._dataframe['filename']):
            wav_len = 160000
            n_vectors = (((wav_len - self.n_fft) // self.hop_size + 1) -
                         self.n_frames) // self.hop_frames + 1
            self.wav_frames.append(n_vectors)
            self.index_to_file.extend([index] * n_vectors)
            self.index_to_chunkindex.extend(range(n_vectors))
        self._len = sum(self.wav_frames)
        self.target_dataframe = self._dataframe[self._dataframe.domain ==
                                                'target']
        self.target_len = len(self.target_dataframe)

    def __del__(self):
        if self.datacache != None:
            self.datacache.close()

    def __len__(self):
        return self._len

    def __getitem__(self, index: int):
        f_index = self.index_to_file[index]
        filename, machinetype, section, domain, label = self._dataframe.iloc[
            f_index][['filename', 'machinetype', 'section', 'domain', 'label']]
        chunkindex = self.index_to_chunkindex[index]
        chunk_beginindex = chunkindex * self.hop_frames
        if self.datacache is None:
            self.datacache = h5py.File(self.h5path, 'r')
        data = self.datacache[filename][chunk_beginindex *
                                        self.hop_size:chunk_beginindex *
                                        self.hop_size + self.chunk_length]
        target = torch.tensor([0.])
        if label != 'normal':
            target = torch.tensor([1.])
        return torch.from_numpy(data), target, domain, section


class HDF5_evalset(tdata.Dataset):
    def __init__(
        self,
        hdf5_path,
        data_frame,
        machine_type,
        domain,
        section=[0, 1, 2],
        acoustic_paras={
            'n_frames': 5,
            'hop_frames': 1,
            'n_mels': 128,
            'hop_size': 512,
            'n_fft': 1024
        },
    ) -> None:
        super(HDF5_evalset, self).__init__()
        self._machinetype = machine_type
        self.section = section
        self.domain = domain
        self._dataframe = data_frame[
            (data_frame.machinetype == self._machinetype)
            & (data_frame.section.isin(self.section)) &
            (data_frame.domain.isin(self.domain))].reset_index()
        self.n_frames = acoustic_paras.get('n_frames', 5)
        self.n_mels = acoustic_paras.get('n_mels', 128)
        self.hop_size = acoustic_paras.get('hop_size', 512)
        self.n_fft = acoustic_paras.get('n_fft', 1024)
        self.hop_frames = acoustic_paras.get('hop_frames', 1)
        self.dims = self.n_frames * self.n_mels
        self.datacache = None
        self.h5path = hdf5_path
        self.chunk_length = self.n_fft + (self.n_frames - 1) * self.hop_size
        self._len = len(self._dataframe)

    def __del__(self):
        if self.datacache != None:
            self.datacache.close()

    def __len__(self):
        return self._len

    def __getitem__(self, index: int):
        filename, machinetype, section, domain, label = self._dataframe.iloc[
            index][['filename', 'machinetype', 'section', 'domain', 'label']]
        if self.datacache is None:
            self.datacache = h5py.File(self.h5path, 'r')
        data = librosa.util.frame(self.datacache[filename][:],
                                  frame_length=self.chunk_length,
                                  hop_length=self.hop_size * self.hop_frames,
                                  axis=0)
        target = torch.tensor([0.])
        if label != 'normal':
            target = torch.tensor([1.])
        return torch.from_numpy(data), target, domain, section


def get_trainloader(hdf5_path, data_frame, machine_type, domain, section,
                    acoustic_paras, **kwargs):
    return tdata.DataLoader(
        HDF5_trainset(hdf5_path, data_frame, machine_type, domain, section,
                      acoustic_paras), **kwargs)


def get_evalloader(hdf5_path, data_frame, machine_type, domain, section,
                   acoustic_paras, **kwargs):
    return tdata.DataLoader(
        HDF5_evalset(hdf5_path, data_frame, machine_type, domain, section,
                     acoustic_paras), **kwargs)


def get_testloader(hdf5_path, data_frame, machine_type, domain, section,
                   acoustic_paras, **kwargs):
    return tdata.DataLoader(
        HDF5_testset(hdf5_path, data_frame, machine_type, domain, section,
                     acoustic_paras), **kwargs)
