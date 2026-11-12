# Brain Tumour Detection Using CNN (MRI) — Deep Learning Project

This project is a reproducible student implementation inspired by the attached paper:
**“Brain Tumor Detection Using CNN: A Deep Learning Approach for MRI Classification”**
(2025, 6th International Conference on Emerging Technology, INCET).

## What this project does
- Binary classification of MRI images: **Tumour (positive)** vs **No Tumour (negative)**.
- CNN architecture follows the architecture table shown in the supplied paper (two convolution blocks, Dense(96), Dropout, Dense(1)).
- Train/validation split and held-out test evaluation.
- Reports accuracy, precision, recall, F1 and confusion matrix.
- Flask web interface for uploading an MRI image and viewing the model output.

> Educational research prototype only. This is not a medical device and must not be used for diagnosis or treatment decisions.

## Paper architecture
The paper's architecture table specifies:
- Conv2D: output 148 × 148 × 96; 2,688 parameters
- MaxPooling2D: 74 × 74 × 96
- Conv2D: 72 × 72 × 32; 27,680 parameters
- MaxPooling2D: 36 × 36 × 32
- Flatten: 41,472 features
- Dense: 96 units; 3,981,408 parameters
- Dropout: 96 units (no parameters)
- Dense: 1 output; 97 parameters
- Total: 4,011,873 trainable parameters

Input images are resized to 150 × 150 and converted to RGB. The Conv2D parameter counts correspond to 3-channel input and 3 × 3 kernels. The final sigmoid output is for binary classification. Paper-reported results (82% accuracy, precision 0.85, recall 0.84, F1 0.79) are benchmarks from the paper, not guaranteed outcomes of this implementation.

## Dataset
The paper describes a Kaggle MRI dataset with 3,264 images and binary labels. Place data in this layout:

```text
data/
  train/
    no_tumor/       # negative class
    tumor/           # positive class
  test/
    no_tumor/
    tumor/
```

Images may be JPG, JPEG, PNG, or BMP. If you have only one combined dataset, use the same two class folders as `data/all/` and run `python split_dataset.py --source data/all --output data`.

Only use datasets you are authorized to access. Keep patient data private.

## Setup (Windows)
1. Install Python 3.10 or 3.11.
2. Extract this ZIP.
3. Open terminal inside project folder.
4. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. Install:
   ```powershell
   pip install -r requirements.txt
   ```

## Train
With `data/train` and `data/test` structured as above:
```powershell
python train.py
```
Training saves:
- `artifacts/brain_tumor_cnn.keras`
- `artifacts/class_names.json`
- `artifacts/training_history.png`

Default: 20 epochs, batch size 32, image size 150×150. Modify `config.py` for settings.

## Evaluate
```powershell
python evaluate.py
```
Creates `artifacts/evaluation_report.txt` and `artifacts/confusion_matrix.png`.

## Predict one image
```powershell
python predict.py --image path\to\mri.jpg
```

## Run the web app
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in your browser and upload an image.

## Project architecture
See `architecture.png`. `model.py` builds the CNN.

## Folder structure
```text
brain_tumor_cnn_project/
  app.py
  config.py
  model.py
  train.py
  evaluate.py
  predict.py
  split_dataset.py
  requirements.txt
  templates/index.html
  static/style.css
  architecture.png
  README.md
```

## Important notes / troubleshooting
- Training needs the MRI dataset; it is not bundled because datasets can be large and licensing varies.
- If GPU is unavailable, TensorFlow trains on CPU but may take longer.
- Dataset directory names are expected to be `no_tumor` and `tumor`; class mapping is written by training.
- If you encounter memory issues, lower `BATCH_SIZE` in `config.py`.
- Dataset leakage can inflate metrics. Keep patient-wise separation if patient identifiers are available.
