# MOT_yolov8

### Setup
```
pip install -r requirements.txt
pip install cython
pip install cython_bbox
```
### Setup evaluation
Go into directory 'eval using trackeval'
```
pip install -r requirements.txt
```
### Download pretrained model
1. ReID model:
```
gdown --id 1N16RJ_hZGgXg3Ls5DObs3HQUKNbUSBzy
mkdir ./mot/weights/
cp ./working/epoch=5-Val_mAP=0.63159-Val_CMC@rank1=0.96626-Val_CMC@rank5=0.98858.ckpt ./working/mot/weights/vit_base_patch16_224_TransReID.ckpt
```

2. YOLOv8
```
gdown --id 1_oda78B4ZxJ5-LLp6X0fHqxpcs1U6feV
mv ./working/best_latest_version.pt ./mot/weights/
```

### Run code
```
python track.py --source ./data/MOT17/MOT17/train 
--reid-weights ./mot/weights/vit_base_patch16_224_TransReID.ckpt
--yolo-weights ./mot/weights/best_latest_version.pt
--class 0 
--tracking-method strongsort 
--save-txt
```
### Run evaluation
Go into directory 'eval using trackeval'
```
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL StrongSORT --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL True --SEQMAP_FOLDER ./data/trackers/mot_challenge/seqmaps --GT_FOLDER ./data/gt/mot_challenge/ --TRACKERS_FOLDER ./data/trackers/mot_challenge/
```

Output Video Save at "runs\track" dir
