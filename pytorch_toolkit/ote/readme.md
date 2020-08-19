## pytorch_toolkit/object_detection tree
```bash
├── face-detection
│   ├── face-detection-0200
│   │   ├── config.py
│   │   └── template.yml
│   ├── face-detection-0202
│   │   ├── config.py
│   │   └── template.yml
│   ├── problem.yml
│   ├── readme.md
│   ├── schema.json
│   └── tools
│       ├── eval.py
│       └── train.py
├── horizontal-text-detection
│   ├── horizontal-text-detection-0001
│   │   ├── config.py
│   │   └── template.yml
│   ├── problem.yml
│   ├── readme.md
│   ├── schema.json
│   └── tools
│       ├── eval.py
│       └── train.py
├── person-detection
│   ├── person-detection-0200
│   │   ├── config.py
│   │   └── template.yml
│   ├── person-detection-0201
│   │   ├── config.py
│   │   └── template.yml
│   ├── person-detection-0202
│   │   ├── config.py
│   │   └── template.yml
│   ├── problem.yml
│   ├── readme.md
│   ├── schema.json
│   └── tools
│       ├── eval.py
│       └── train.py
├── oteod
├── requirements.txt
├── setup.py
```

## template.yml
```bash
name: person-vehicle-bike-detection-2001
description: Person Vehicle Bike Detection based on MobileNetV2 (SSD).
dependencies:
- sha256: 9c4190208d9ff7b7b860821c48264a50ee529e84f23d0d2f6947eceb64c2346a
  size: 14937205
  source: https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/vehicle-person-bike-detection-2001-1.pth
  destination: snapshot.pth
- source: ../tools/train.py
  destination: train.py
- source: ../tools/eval.py
  destination: eval.py
- source: ../../tools/export.py
  destination: export.py
- source: ../../tools/quantize.py
  destination: quantize.py
training_parameters:
  gpu_num: 4
  batch_size: 54
  base_learning_rate: 0.05
  epochs: 20
metrics:
- display_name: Size
  key: size
  unit: Mp
  value: 1.84
- display_name: Complexity
  key: complexity
  unit: GFLOPs
  value: 1.86
- display_name: mAP @ [IoU=0.50:0.95]
  key: map
  unit: '%'
  value: 22.6
config: config.py
```

## train.py
```bash
python ../tools/train.py -h
usage: train.py [-h] --train_ann_files TRAIN_ANN_FILES --train_img_roots
                TRAIN_IMG_ROOTS --val_ann_files VAL_ANN_FILES --val_img_roots
                VAL_IMG_ROOTS [--resume_from RESUME_FROM]
                [--load_weights LOAD_WEIGHTS]
                [--save_checkpoints_to SAVE_CHECKPOINTS_TO] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--gpu_num GPU_NUM]
                [--base_learning_rate BASE_LEARNING_RATE] [--config CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  --train_ann_files TRAIN_ANN_FILES
                        Comma-separated paths to training annotation files.
  --train_img_roots TRAIN_IMG_ROOTS
                        Comma-separated paths to training images folders.
  --val_ann_files VAL_ANN_FILES
                        Comma-separated paths to validation annotation files.
  --val_img_roots VAL_IMG_ROOTS
                        Comma-separated paths to validation images folders.
  --resume_from RESUME_FROM
                        Resume training from previously saved checkpoint
  --load_weights LOAD_WEIGHTS
                        Load only weights from previously saved checkpoint
  --save_checkpoints_to SAVE_CHECKPOINTS_TO
                        Location where checkpoints will be stored
  --epochs EPOCHS       Number of epochs during training
  --batch_size BATCH_SIZE
                        Size of a single batch during training per GPU.
  --gpu_num GPU_NUM     Number of GPUs that will be used in training, 0 is for
                        CPU mode.
  --base_learning_rate BASE_LEARNING_RATE
                        Starting value of learning rate that might be changed
                        during training according to learning rate schedule
                        that is usually defined in detailed training
                        configuration.
  --config CONFIG       Location of a file describing detailed model
                        configuration.
```