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

## Usage for pickle files (Training))
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

