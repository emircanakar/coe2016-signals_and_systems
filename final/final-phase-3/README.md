# Emo Challenge 2026 - Emotion Classification Project

This repository contains the necessary scripts for extracting audio features, training a Support Vector Machine (SVM) model, and running a live emotion classification demo.

## Execution Order

To correctly execute the pipeline, you must run the files in the strict order below.

### Step 1: Feature Extraction (`process.py`)

This script processes the raw audio files and extracts advanced acoustic features (MFCCs, ZCR, RMS, etc.).

- **Important:** Before running this script, you must open `process.py` and update the `DATASET_DIR` variable to point to the exact path of the dataset directory on your local machine.
- **Output:** Generates the `advanced_features.csv` file, which is required for the next step.

### Step 2: Model Training (`model-train.py`)

Once the CSV file is ready, run this script to train the SVM classifier and scale the data.

- **Requirement:** Needs `advanced_features.csv` to be present in the same directory.
- **Output:** Generates the serialized model weights and scaler parameters as `svm_model.pkl` and `scaler.pkl`.

### Step 3: Live Demo (`demo.py`)

This is the interactive/live demonstration script. It analyzes a selected audio file and outputs the predicted emotion along with class probabilities.

- **Requirement:** Needs both `svm_model.pkl` and `scaler.pkl` to function.

---

## Visualization Scripts

The following scripts are solely used for generating statistical plots and visualizations for the presentation/report. They do not affect the main pipeline and can be run independently once the required data is available:

- `final_mfcc_comparison.py`
- `scatter-boxplot-comp.py`
- `time_domain.py`
