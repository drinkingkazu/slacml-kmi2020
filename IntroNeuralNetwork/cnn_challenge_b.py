from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
import torch, h5py
import numpy as np
from torch.utils.data import Dataset

class SegmentImage2D(Dataset):

    def __init__(self, data_files, num_class=3, start=0.0, end=1.0, normalize=None):
        
        # Validate file paths
        self._files = [f if f.startswith('/') else os.path.join(os.getcwd(),f) for f in data_files]
        for f in self._files:
            if os.path.isfile(f): continue
            sys.stderr.write('File not found:%s\n' % f)
            raise FileNotFoundError
            
        if start < 0. or start > 1.:
            print('start must take a value between 0.0 and 1.0')
            raise ValueError
        
        if end < 0. or end > 1.:
            print('end must take a value between 0.0 and 1.0')
            raise ValueError
            
        if end <= start:
            print('end must be larger than start')
            raise ValueError

        # Loop over files and scan events
        self._file_handles = [None] * len(self._files)
        self._entry_to_file_index  = []
        self._entry_to_data_index = []
        self._shape = None
        self.classes = np.arange(num_class)
        self._normalize = normalize
        for file_index, file_name in enumerate(self._files):
            f = h5py.File(file_name,mode='r',swmr=True)
            # data size should be common across keys (0th dim)
            data_size = f['image'].shape[0]
            self._entry_to_file_index += [file_index] * data_size
            self._entry_to_data_index += range(data_size)
            
            f.close()
            
        self._start  = int(len(self._entry_to_file_index)*start)
        self._length = int(len(self._entry_to_file_index)*end) - self._start

    def __del__(self):
        for i in range(len(self._file_handles)):
            if self._file_handles[i]:
                self._file_handles[i].close()
                self._file_handles[i]=None 


    def __len__(self):
        return self._length
    
    def __getitem__(self,idx):
        file_index  = self._entry_to_file_index[self._start+idx]
        entry_index = self._entry_to_data_index[self._start+idx]
        if self._file_handles[file_index] is None:
            self._file_handles[file_index] = h5py.File(self._files[file_index],mode='r',swmr=True)

        fh = self._file_handles[file_index]

        image = torch.Tensor(fh['image'][entry_index])
        label = torch.Tensor(fh['label'][entry_index])
        if self._normalize:
            image = (image - self._normalize[0])/self._normalize[1]

        return dict(data=image,label=label,index=self._start+idx)

    
def collate(batch):
    res = dict(data  = torch.cat([sample['data'][None][None] for sample in batch],0),
               label = torch.cat([sample['label'][None] for sample in batch],0).long(),
               index = np.array([sample['index'] for sample in batch]),
              )
    return res
    
