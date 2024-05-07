import more_itertools
from typing import Any, Dict, Iterable, Union, List, Mapping
import vqa_data
import refcoco_data
import itertools
import random

class MultitaskLoader(object):
    def __init__(self, loaders, shuffle=True, drop_last=False, sampling='roundrobin', n_batches=None, verbose=True):
        self.loaders = loaders
        self.verbose = verbose
        # self.loader_lens = [len(loader) for loader in self.loaders]
        self.task2len = {loader.task: len(loader) for loader in self.loaders}
        if self.verbose:
            print('Task2len:', self.task2len)
        self.task2loader = {loader.task: loader for loader in self.loaders}
        # print('loader lens:', self.loader_lens)

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampling = sampling
        self.epoch_tasks = None
        self.n_batches = n_batches
        self.set_epoch(0)
        # print('loader indices:', self.loader_indices)

    def __iter__(self):
        self.task2iter = {loader.task: iter(loader) for loader in self.loaders}
        # self.loader_iters = [iter(loader) for loader in self.loaders]

        return self

    def set_epoch(self, epoch):
        for loader in self.loaders:
            loader.sampler.set_epoch(epoch)

        if self.sampling == 'roundrobin':
            epoch_tasks = []
            for task, loader in self.task2loader.items():
                n_batches = len(loader)
                epoch_tasks.extend([task]*n_batches)
        elif self.sampling == 'balanced':
            if self.n_batches is None:
                n_batches = sum(self.task2len.values()) // len(self.loaders)
            else:
                n_batches = self.n_batches
            if self.verbose:
                print('# batches:', n_batches)
            epoch_tasks = []
            for task, loader in self.task2loader.items():
                epoch_tasks.extend([task]*n_batches)

        if self.shuffle:
            random.Random(epoch).shuffle(epoch_tasks)
        self.epoch_tasks = epoch_tasks
        if self.verbose:
            print('# epoch_tasks:', len(self.epoch_tasks))

    def __next__(self):
        if len(self.epoch_tasks) > 0:
            task = self.epoch_tasks.pop()
            loader_iter = self.task2iter[task]
            return next(loader_iter)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.epoch_tasks)



def _chunked_iterator(i: Iterable, chunk_size: int, drop_last: bool):
    chunks = more_itertools.chunked(i, chunk_size)
    if drop_last:
        return (chunk for chunk in chunks if len(chunk) == chunk_size)
    else:
        return chunks
