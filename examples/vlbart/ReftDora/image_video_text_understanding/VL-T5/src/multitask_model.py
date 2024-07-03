
from modeling_t5 import VLT5

from vqa_model import VLT5VQA
from gqa_model import VLT5GQA
from nlvr_model import VLT5NLVR
from refcoco_model import VLT5RefCOCO
from caption_model import VLT5COCOCaption
from mmt_model import VLT5MMT
from vcr_model import VLT5VCR
from classification_model import VLT5Classification

class VLT5MultiTask(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLT5VQA.train_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLT5GQA.train_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.train_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLT5RefCOCO.train_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.train_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLT5MMT.train_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLT5VCR.train_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLT5Classification.train_step(self, batch, **kwargs)

    def valid_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLT5VQA.valid_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLT5GQA.valid_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.valid_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLT5RefCOCO.valid_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.valid_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLT5MMT.valid_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLT5VCR.valid_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLT5Classification.valid_step(self, batch, **kwargs)

    def test_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLT5VQA.test_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLT5GQA.test_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLT5NLVR.test_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLT5RefCOCO.test_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLT5COCOCaption.test_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLT5MMT.test_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLT5VCR.test_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLT5Classification.test_step(self, batch, **kwargs)


from modeling_bart import VLBart

from vqa_model import VLBartVQA
from gqa_model import VLBartGQA
from nlvr_model import VLBartNLVR
from refcoco_model import VLBartRefCOCO
from caption_model import VLBartCOCOCaption
from mmt_model import VLBartMMT
from vcr_model import VLBartVCR
from classification_model import VLBartClassification

class VLBartMultiTask(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLBartVQA.train_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLBartGQA.train_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLBartNLVR.train_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLBartRefCOCO.train_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLBartCOCOCaption.train_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLBartMMT.train_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLBartVCR.train_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLBartClassification.train_step(self, batch, **kwargs)

    def valid_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLBartVQA.valid_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLBartGQA.valid_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLBartNLVR.valid_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLBartRefCOCO.valid_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLBartCOCOCaption.valid_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLBartMMT.valid_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLBartVCR.valid_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLBartClassification.valid_step(self, batch, **kwargs)

    def test_step(self, batch, **kwargs):
        task = batch['task']
        if task == 'vqa':
            return VLBartVQA.test_step(self, batch, **kwargs)
        elif task == 'gqa':
            return VLBartGQA.test_step(self, batch, **kwargs)
        elif task == 'nlvr':
            return VLBartNLVR.test_step(self, batch, **kwargs)
        elif task == 'refcoco':
            return VLBartRefCOCO.test_step(self, batch, **kwargs)
        elif task == 'caption':
            return VLBartCOCOCaption.test_step(self, batch, **kwargs)
        elif task == 'mmt':
            return VLBartMMT.test_step(self, batch, **kwargs)
        elif task == 'vcr':
            return VLBartVCR.test_step(self, batch, **kwargs)
        elif task == 'cls':
            return VLBartClassification.test_step(self, batch, **kwargs)

