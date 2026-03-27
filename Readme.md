---

# 🌾 CropVision: Satellite-Based Crop Health Monitoring using Deep Learning

## 📌 Overview

CropVision is an AI-based system that analyzes **multi-spectral satellite imagery** to assess crop health at a spatial level. It uses vegetation indices (NDVI, SAVI, NDWI) and a **CNN-based model** to classify farmland into health categories.

Unlike basic agriculture ML projects, this system operates on **GeoTIFF raster data** and performs **patch-wise inference** to generate full-field health maps.

---

## 🎯 Key Contributions

* Built a **geospatial ML pipeline** using satellite-derived vegetation indices
* Implemented **patch-based CNN inference (64×64 sliding window)** for large images
* Developed an **interactive Streamlit dashboard** for visualization
* Integrated **TensorFlow/Keras model** with real inference pipeline
* Designed a **multi-location analysis system** (Ludhiana, Ujjain, Thanjavur)

---

## ⚙️ Tech Stack

**Machine Learning**

* TensorFlow / Keras (CNN model)
* NumPy

**Geospatial Processing**

* Rasterio (GeoTIFF handling)
* NDVI, SAVI, NDWI index processing

**Visualization**

* Streamlit (dashboard)
* Matplotlib, Plotly

**Backend**

* Flask (basic inference API)

---

## 🏗️ Project Structure

```
crop_monitoring_sih/
│
├── data/
│   ├── raw/                      # Satellite data archives (.zip)
│   └── processed/
│       ├── Ludhiana/
│       ├── Thanjavur/
│       └── Ujjain/
│           ├── NDVI.tif
│           ├── NDWI.tif
│           ├── SAVI.tif
│           └── EVI.tif
│
├── model/
│   └── crop_health_model.keras
│
├── notebooks/
│   ├── Build_and_Split_Dataset.ipynb
│   └── Evaluating_Satellite_Data.ipynb
│
├── app.py                       # Flask API
├── dashboard.py                # Basic Streamlit dashboard
├── cropvision_dashboard.py     # Advanced dashboard (multi-tab UI)
├── requirements.txt
└── .github/workflows/
```

---

## 🔄 System Workflow

### 1. Data Preparation

* Satellite imagery converted into vegetation indices:

  * NDVI (vegetation health)
  * SAVI (soil-adjusted vegetation)
  * NDWI (water content)

### 2. Input Representation

* Indices normalized and stacked into a **3-channel image tensor**

### 3. Patch-Based Inference

* Image divided into **64×64 patches**
* Model predicts each patch
* Predictions reconstructed into full spatial map

### 4. Output Classes

* High Stress
* Moderate Stress
* Healthy

### 5. Visualization

* NDVI heatmap (raw satellite data)
* AI-generated crop health map
* Area-wise health distribution (%)

---

## 📊 Model Performance

* Training Accuracy: **~99%**
* Validation Accuracy: **~91–92%**

### Observations

* Model learns strong spatial features from vegetation indices
* Validation accuracy stabilizes early (~epoch 4–5)
* Slight overfitting observed as training loss continues decreasing

---

## 🧠 Interpretation

The model demonstrates effective feature extraction from multi-spectral inputs but is constrained by:

* Limited dataset diversity
* Lack of temporal variation

---

## 📊 Dashboard Features

### 🌱 Crop Health Mapping

* NDVI visualization vs AI prediction map
* Color-coded classification (stress levels)

### 📈 Farm Health Summary

* Percentage distribution of:

  * Healthy
  * Moderate stress
  * High stress

### 🌍 Soil Condition (Simulated)

* Randomized soil metrics for visualization

### 🐛 Pest Risk Estimation

* Rule-based system using:

  * Crop stress %
  * Simulated weather data

### 📊 Multi-Tab Interface

* Crop Health
* Soil Condition
* Pest Risk
* Summary

---

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run dashboard

```bash
streamlit run cropvision_dashboard.py
```

### Run backend API

```bash
python app.py
```

---

## 📈 Strengths

* Works on **real satellite GeoTIFF data**
* Implements **spatial ML (not tabular)**
* Uses **patch-wise CNN inference for large images**
* Produces **visual, interpretable outputs (maps)**

---

## 🔧 Future Improvements

* Add **EarlyStopping and regularization (Dropout)**
* Perform **cross-validation and report metrics (F1, confusion matrix)**
* Integrate **real-time satellite or IoT data sources**
* Connect Flask API to **actual image-based inference**
* Deploy dashboard (Streamlit Cloud / Render)
* Expand dataset across **multiple regions and seasons**

---

## 🧑‍💻 Author

**Krish Garg**
GitHub: [https://github.com/krishgarg344](https://github.com/krishgarg344)
