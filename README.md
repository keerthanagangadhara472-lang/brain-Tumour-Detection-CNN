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
