
---

# YOLOv8-Based Person and PPE Detection

This project involves training YOLOv8 models to detect persons and Personal Protective Equipment (PPE) in images. The project includes format conversion, model training, and inference.

## Project Overview

- **Person Detection Model**: A YOLOv8 model trained to detect persons in images.
- **PPE Detection Model**: A YOLOv8 model trained on cropped images of detected persons to identify various PPE items (e.g., hard-hats, gloves, boots).

## Steps to Run the Project

### 1. Format Conversion

Convert PascalVOC annotations to YOLOv8 format:




```bash
python pascalVOC_to_yolo.py /path/to/pascalVOC_labels /path/to/output_dir
```

#### Note that the directory containing .xml files should have exact name 'pascalVOC_labels' 
(eg: command  python pascalVOC_to_yolo.py /home/user/dataset/pascalVOC_labels  /path/to/output_dir)

### 2. Model Training

Train the person detection and PPE detection models using YOLOv8. Use the provided training scripts and datasets.

### 3. Inference

Run inference on a directory of images:

```bash
python inference.py --input_dir /path/to/input/images --output_dir /path/to/output/images --person_det_model /path/to/person/model --ppe_detection_model /path/to/ppe/model
```

## Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

- **pascalVOC_to_yolo.py**: Script for converting PascalVOC annotations to YOLOv8 format.
- **inference.py**: Script for running inference with both person and PPE detection models.
- **requirements.txt**: List of required Python packages.
