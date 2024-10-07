# Evaluation code for Cherry CO

## Requirements
Make sure you have the following installed before running the model:
* Python 3.8+
* PyTorch 1.10+
* CUDA 11.3+ (for GPU training)
* Other dependencies in requirements.txt
  
## Installation
First, install the PyTorch (torch), torchvision and torchaudio packages with support for CUDA 12.4
```commandline
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
```

Then, install the basic requirements: 
```commandline
pip install -r requirements
```
## Training the Model
Once your dataset is ready, you can start training YOLOv7.

1. Run the training command:
```commandline
python train_aux.py --workers 8 --device 0 --batch-size 2 --data data/ripeness.yaml --img 1024 1024 --cfg cfg/training/yolov7-e6_ripeness.yaml --weights '1' --name yolov7-e6 --hyp data/hyp.scratch.custom.yaml
#Example
```
Parameters:
* --workers 8: Uses 8 CPU workers for loading data.
* --device 0: Specifies the GPU (device 0) for training.
* --batch-size 2: Processes 2 images per training batch.
* --data data/ripeness.yaml: Path to the dataset configuration file.
* --img 1024 1024: Resizes input images to 1024x1024 pixels.
* --cfg cfg/training/yolov7-e6_ripeness.yaml: Model architecture configuration file (YOLOv7-e6 variant).
* --weights '1': Starts training from scratch (no pre-trained weights).
* --name yolov7-e6: Names the training run yolov7-e6.
* --hyp data/hyp.scratch.custom.yaml: Loads custom hyperparameters for training.
  
2. Training configurations: You can also tweak the hyperparameters and architecture by editing the configuration file (cfg/yolov7.yaml).

## Evaluation
Once the training is completed, you can evaluate the model by viewing the trained model in runs. Inside you will find accuracy metrics such as F1_curve, P_curve, R_curve and also in results.png you will find a compilation of visualizations of metrics such as RECALL, MAP, etc. 

## Usage for pickle files 
- Ripeness:
```commandline
python main.py --config datasets/cherry_instance_ripeness.py --pkl-results work_dir/results.pkl --coco-format --eval proposal
```

## Usage for yolo like folders
- Ripeness:
```commandline
python main.py --config datasets/cherry_instance_ripeness.py --folder-results work_dir/yolov7 --eval proposal
```

## Base de Datos

