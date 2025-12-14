# Fruit-Detection

#How to Run

Install Python 3
https://www.python.org/downloads/
Install required dependencies:
pip install tensorflow opencv-python numpy matplotlib scikit-learn
Organize the dataset:
dataset/
├── train/
│   ├── fresh/
│   └── rotten/
└── test/
    ├── fresh/
    └── rotten/
Train the model:
python3 train_model.py
Run Grad-CAM on an image:
python3 gradcam_viewer.py images/applemoldycore16-1754f.jpg 
This will generate a Grad-CAM visualization and mold coverage estimate.
