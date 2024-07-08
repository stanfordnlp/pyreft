from collections import defaultdict
class TrainingMeter():
    def __init__(self):
        self.counter_dict = defaultdict(float)
        self.true_dict = defaultdict(float)

    def update(self, loss_dict):
        for key, item in loss_dict.items():
            self.counter_dict[key] += 1
            self.true_dict[key] += item

    def report(self, logger = None):
        keys = list(self.counter_dict.keys())
        keys.sort()
        for key in keys:
            if logger is None:
                print("  {} : {:.7}".format(key, self.true_dict[key] / self.counter_dict[key]))
            else:
                logger.info("  {} : {:.7}".format(key, self.true_dict[key] / self.counter_dict[key]))
    
    def clean(self):
        self.counter_dict = defaultdict(float)
        self.true_dict = defaultdict(float)


from lz4.frame import compress, decompress
from collections import defaultdict
from contextlib import contextmanager
import io
import json
from os.path import exists
import msgpack
import msgpack_numpy
import collections
import lmdb
msgpack_numpy.patch()

class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True, readahead=False):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False,
                                 readahead=readahead)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret
