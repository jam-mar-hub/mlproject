# 🧬 AML Survival Prediction (End-to-End Machine Learning)

This project is a complete **End-to-End Machine Learning solution** designed to predict the survival probability of patients diagnosed with **Acute Myeloid Leukemia (AML)**.

It encompasses the entire data science lifecycle: from data ingestion and cleaning to model training and deployment via a Flask Web Application.

## 📋 Project Overview

The goal is to assist medical professionals in evaluating patient prognosis based on specific clinical biomarkers and genetic mutations. Unlike standard regression tasks, this project utilizes **Survival Analysis** techniques to handle time-to-event data and censored observations.

### Key Features
* **Complex Data Merging:** Integrates Clinical data, Molecular (genetic) data, and Target (survival) data.
* **Feature Engineering:** Automatic calculation of the mutation burden (`Nmut`) per patient.
* **Robust Pipeline:** Modular code structure separating data ingestion, transformation, and training.
* **Survival Model:** Implementation of the **Random Survival Forest** algorithm using `scikit-survival`.
* **Web Interface:** A user-friendly **Flask** application for real-time predictions.

## 🛠️ Technical Architecture

The project follows industry-standard software engineering practices with a modular structure:

```text
├── artifacts/          # Stores generated models (.pkl) and processed CSVs
├── notebook/           # Jupyter notebooks and raw data
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Reads, merges, and splits raw data
│   │   ├── data_transformation.py # Preprocessing (OneHotEncoding, Scaling, Sparse fix)
│   │   └── model_trainer.py       # Trains the Random Survival Forest
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Orchestrator to run the full training workflow
│   │   └── predict_pipeline.py    # Logic for generating predictions in the app
│   └── utils.py                   # Utility functions (save/load objects)
├── templates/          # HTML files for the web app
├── app.py              # Flask Application entry point
└── requirements.txt    # Project dependencies