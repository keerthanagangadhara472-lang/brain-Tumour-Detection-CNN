<<<<<<< HEAD
# Brain Tumour Detection Using CNN

A beginner-friendly educational project that classifies brain MRI images into **tumour** and **no-tumour** classes using a Convolutional Neural Network (CNN), TensorFlow/Keras and Streamlit.

> **Important:** This is an academic prototype, not a medical device. It must not be used to diagnose, rule out, or guide treatment for a real patient. Clinical decisions require qualified medical professionals and validated systems.

## Project structure

```text
Brain_Tumour_Detection_CNN/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── train.py
│   └── predict.py
├── data/
│   ├── train/
│   │   ├── no_tumor/
│   │   └── tumor/
│   └── test/
│       ├── no_tumor/
│       └── tumor/
├── models/                 # trained model is generated here
└── outputs/                # plots and evaluation report are generated here
```

## 1. Requirements

- Python 3.10 or 3.11 recommended
- VS Code or another Python IDE
- Internet connection to install packages and download a permitted MRI dataset
- 8 GB RAM recommended; training time depends on hardware and dataset size

## 2. Install and activate environment

Open the project folder in VS Code terminal.

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Prepare dataset

Put images into the four folders shown below:

```text
data/train/no_tumor/
data/train/tumor/
data/test/no_tumor/
data/test/tumor/
```

Use a labelled MRI dataset whose licence permits your intended academic use. Keep test images separate from training images. Prefer splitting by patient rather than randomly splitting individual slices from the same patient.

## 4. Train CNN

From the project root:
```bash
python src/train.py
```

The script resizes images to 150×150, normalizes pixel values, trains the CNN, saves the best model to `models/brain_tumour_cnn.keras`, and writes training curves, confusion matrix and a classification report to `outputs/`.

## 5. Predict one image

```bash
python src/predict.py path/to/your_mri_image.jpg
```

## 6. Run the web application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, upload an MRI image, and click **Analyse MRI**. The app requires the trained model.

## CNN architecture

1. Input: 150×150×3 MRI image
2. Conv2D: 16 filters, 3×3, ReLU
3. MaxPooling2D: 2×2
4. Conv2D: 32 filters, 3×3, ReLU
5. MaxPooling2D: 2×2
6. Flatten
7. Dense: 96 neurons, ReLU
8. Dropout: 0.5
9. Output: 1 neuron, sigmoid

Binary cross-entropy loss and Adam optimizer are used. A 0.5 threshold separates the two model classes.

## Evaluation

Use the held-out test set. Review accuracy, precision, recall, F1-score and confusion matrix; do not report training accuracy as test performance. Actual results depend on the dataset, split, training settings and hardware. No accuracy is guaranteed.

## GitHub upload

1. Create a new empty repository on GitHub, for example `Brain-Tumour-Detection-CNN`.
2. Open terminal in this project folder.
3. Run:

```bash
git init
git add .
git commit -m "Add brain tumour detection CNN project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Brain-Tumour-Detection-CNN.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If Git asks you to sign in, complete the browser authentication. Do not put passwords or access tokens inside source code.

### Uploading later changes

```bash
git add .
git commit -m "Update CNN project"
git push
```

### Important GitHub notes

- The `.gitignore` excludes image datasets, trained model files and generated plots to avoid pushing large or sensitive files.
- Add a dataset source/citation and licence information to your report/README.
- Do not upload identifiable patient scans unless you have permission and a lawful basis.
- GitHub stores your source code; it does not automatically train or run the model.

## Reference paper

The supplied 4-page paper, *Brain Tumor Detection Using CNN: A Deep Learning Approach for MRI Classification*, is the starting reference for the CNN layer sequence. Its dataset descriptions and reported metrics are not fully consistent across pages. This implementation therefore uses a clear folder-based dataset workflow and generates fresh evaluation results when you train it.
=======
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
>>>>>>> 12946e920a2f7d36a126a0784b7d2f9b384c55fc
