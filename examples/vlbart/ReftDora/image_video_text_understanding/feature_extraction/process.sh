# wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip
# wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip
unzip train2014_obj36.zip -d .
unzip val2014_obj36.zip -d .
python tsv_to_h5.py --tsv_path train2014_obj36.tsv --h5_path train2014_obj36.h5
python tsv_to_h5.py --tsv_path val2014_obj36.tsv --h5_path val2014_obj36.h5
# Get resplit_val_obj36.h5 from val2014_obj36.h5
python coco_val_compact.py

# Pretrain(VG)/GQA: Download LXMERT's VG features (tsv) and convert to hdf5
# wget https://nlp.cs.unc.edu/data/lxmert_data/vg_gqa_imgfeat/vg_gqa_obj36.zip
unzip vg_gpa_obj36.zip -d .
python tsv_to_h5.py --tsv_path vg_gqa_obj36.tsv --h5_path vg_gqa_obj36.h5

# RefCOCOg
python refcocog_gt.py --split train
python refcocog_mattnet.py --split val
python refcocog_mattnet.py --split test

# NLVR2: Download LXMERT's COCO features (tsv) and convert to hdf5
# wget https://nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/train_obj36.zip
# wget https://nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/valid_obj36.zip
# wget https://nlp.cs.unc.edu/data/lxmert_data/nlvr2_imgfeat/test_obj36.zip
unzip train_obj36.zip -d .
unzip valid_obj36.zip -d .
unzip test_obj36.zip -d .

python tsv_to_h5.py --tsv_path train_obj36.tsv --h5_path train_obj36.h5
python tsv_to_h5.py --tsv_path valid_obj36.tsv --h5_path valid_obj36.h5
python tsv_to_h5.py --tsv_path test_obj36.tsv --h5_path test_obj36.h5

# Multi30K
# Download images following https://github.com/multi30k/dataset
python flickr30k_proposal.py --split trainval
python flickr30k_proposal.py --split test2017
python flickr30k_proposal.py --split test2018
