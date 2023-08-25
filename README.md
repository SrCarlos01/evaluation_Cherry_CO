# Evaluation code for Cherry CO


## Installation

First install the basic requirements: 
```commandline
pip install -r requirements
```

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
