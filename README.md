# Crack Detection Desktop Suite

## Overview
This repository implements a Kivy/KivyMD desktop application for training and evaluating convolutional neural networks that detect structural cracks. The dashboard allows you to:

- Select a dataset directory that mirrors the Google Drive folder structure you shared (class subdirectories containing images).
- Launch a dedicated training workspace that sequentially fits four transfer-learning models (VGG16, VGG19, ResNet50, EfficientNetB0) while streaming progress bars and logs.
- Persist trained weights, training history, and artefacts directly beside the dataset for full reproducibility.
- Open a prediction workspace that loads the trained weights, performs inference on user-supplied images, and visualises the results through confusion matrices, Grad-CAM grids, calibration curves, UMAP embeddings, probability histograms, and duplicate/leakage plots.

All heavy computation is performed off the UI thread so the interface stays responsive. Model evaluation metadata is cached to speed up successive sessions.

## Repository Layout
```
CNN-project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Place the downloaded Google Drive dataset here (class folders)
│   ├── processed/        # Cached tensors created by the preprocessing pipeline
│   └── metadata/         # CSVs with label maps, duplicate reports, etc.
├── models/
│   ├── checkpoints/      # Saved model checkpoints (.keras)
│   ├── exported/         # Optional ONNX/TFLite exports
│   └── history/          # Training history JSON logs
├── src/
│   ├── app/              # Kivy/KivyMD application package
│   │   ├── main.py
│   │   ├── screens.py
│   │   ├── widgets.py
│   │   ├── state.py
│   │   └── kv/
│   │       ├── dashboard.kv
│   │       ├── training.kv
│   │       └── inference.kv
│   ├── config/
│   │   ├── settings.py
│   │   └── logging.yaml
│   ├── data_pipeline/
│   │   ├── loader.py
│   │   ├── augment.py
│   │   └── hashing.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── vgg16.py
│   │   ├── vgg19.py
│   │   ├── resnet50.py
│   │   └── efficientnet.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── visualizations.py
│   │   └── reporting.py
│   └── utils/
│       ├── file_io.py
│       ├── logging_utils.py
│       └── threading.py
└── tests/
    ├── test_data_pipeline.py
    ├── test_model_factories.py
    └── test_visualizations.py
```

## Getting Started

### 1. Download the Dataset
1. Visit the [Google Drive folder](https://drive.google.com/drive/folders/1KmAZNDvGN1IIwcR8mhZYq6MUJKPKCSIz?usp=drive_link).
2. Download the entire folder and place its contents inside `data/raw/`. The structure should resemble `data/raw/<class_name>/*.jpg`.
3. If you have an accompanying CSV with labels/metadata, store it under `data/metadata/`.

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python -m src.app.main
```
The first time you run the app it will prompt for the dataset directory. Subsequent runs remember the last-used directory and output location.

## Training Workflow
1. From the dashboard select **"Open Training Workspace"**.
2. Choose the dataset folder and output directory. The default output path mirrors the dataset root.
3. Optionally adjust hyperparameters (batch size, epochs, learning rate) before starting.
4. Press **Start Training** to sequentially train VGG16, VGG19, ResNet50, and EfficientNetB0.
5. Real-time progress bars, log entries, and sample Grad-CAM thumbnails are displayed per model.
6. On completion, checkpoints are stored in `models/checkpoints/<model_name>/timestamp/` and history JSONs in `models/history/`.

## Prediction & Visualisation Workflow
1. From the dashboard, open the **Prediction Workspace** and pick your trained checkpoints directory.
2. Upload one or more images. The pipeline automatically mirrors the training preprocessing steps.
3. Predictions from each model appear in a comparison table along with confidence scores.
4. Visualisation tabs display:
   - Confusion matrix (aggregated over the current session)
   - Probability histogram with calibration curve overlay
   - UMAP embedding of feature vectors
   - Grad-CAM grid showing heatmaps for each model
   - Duplicate/leakage detection plots based on perceptual hash distances
5. Use the export button to save combined reports as PNG or HTML.

## Configuration
- Default hyperparameters live in `src/config/settings.py`.
- Logging verbosity and formatting are handled by `src/config/logging.yaml`.
- UI theming can be tweaked via the `.kv` files or `kivymd` theme settings in `src/app/main.py`.

## Testing
Basic unit tests reside in the `tests/` directory. Execute them with:
```bash
pytest
```

## Troubleshooting
- **Kivy install issues**: ensure you have the appropriate platform dependencies (SDL2, GStreamer). Refer to the [Kivy installation docs](https://kivy.org/doc/stable/installation/installation.html).
- **TensorFlow GPU support**: install the matching CUDA/cuDNN toolkits. Consult the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu).
- **Large dataset handling**: adjust `CACHE_SIZE` and `AUTOTUNE` parameters in `data_pipeline/loader.py` if you encounter memory constraints.

## Roadmap
- Add experiment tracking (Weights & Biases integration).
- Provide optional mixed-precision training for GPU acceleration.
- Package the application as a standalone executable with PyInstaller.

## License
This project is distributed for personal experimentation. Please ensure you have the rights to use the dataset before training models.
