
<div align="center">

<img src="assets/MLStudio.png" width="130" alt="Machine Learning Studio Logo">

# Machine Learning Studio

### Explore Data · Discover Patterns · Predict Outcomes

An interactive desktop application for exploring datasets,
applying Machine Learning models, evaluating performance,
and visualizing analytical results.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D4?style=for-the-badge&logo=windows&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-805AD5?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-2EA44F?style=for-the-badge)

</div>

---

## 📖 Project Overview

**Machine Learning Studio** is a Python-based desktop application
that brings Machine Learning, data exploration, predictive
analytics, model evaluation, and data visualization together
within a unified graphical environment.

The application is designed to make Machine Learning concepts
more accessible through interactive tools, pre-trained models,
real-world datasets, and comprehensive analytical insights.

Users can explore datasets, compare algorithms, generate
predictions, evaluate model performance, and discover meaningful
patterns without writing code.

---

## 📊 Project at a Glance

| Category | Details |
| :--- | :--- |
| Application Type | Desktop Application |
| Machine Learning Tasks | 3 |
| Integrated Datasets | 31 |
| Regression Datasets | 12 |
| Classification Datasets | 16 |
| Clustering Datasets | 3 |
| User Interface | Interactive GUI |
| Appearance | Light & Dark Mode |
| Supported Platform | Windows |
| Distribution | Standalone Executable |
| Development Status | Completed |

---

## ✨ Key Features

- Modern and interactive graphical user interface
- Light and dark appearance modes
- Exploration of 31 integrated datasets
- Regression, Classification, and Clustering
- Interactive prediction using pre-trained models
- Dynamic feature input controls
- Comprehensive model performance evaluation
- Machine Learning algorithm comparison
- Recommended model identification
- Feature importance analysis
- Correlation matrix visualization
- Interactive analytical charts
- Organized dataset and model management
- Standalone Windows executable

---

## 🧠 Machine Learning Algorithms

### 📈 Regression

Regression algorithms are used to predict continuous numerical
values and analyze relationships between input features and
target variables.

| Algorithm | Status |
| :--- | :---: |
| Linear Regression | ✅ |
| Ridge Regression | ✅ |
| Random Forest Regressor | ✅ |
| Gradient Boosting Regressor | ✅ |
| Support Vector Regression (SVR) | ✅ |

### 🎯 Classification

Classification algorithms are used to predict categorical
outcomes and assign observations to predefined classes.

| Algorithm | Status |
| :--- | :---: |
| Logistic Regression | ✅ |
| Decision Tree | ✅ |
| Random Forest | ✅ |
| K-Nearest Neighbors (KNN) | ✅ |
| Support Vector Machine (SVM) | ✅ |
| Naive Bayes | ✅ |

### 🧩 Clustering

Clustering is an unsupervised learning approach used to discover
hidden patterns and group similar observations within datasets.

| Algorithm | Status |
| :--- | :---: |
| K-Means Clustering | ✅ |

---

## 🗂️ Integrated Datasets

The application includes 31 datasets covering various domains,
including healthcare, finance, environmental science, energy,
engineering, business, and social analytics.

### 📈 Regression — 12 Datasets

| ID | Dataset |
| :---: | :--- |
| D01 | Abalone |
| D03 | Real Estate Valuation |
| D08 | Auto MPG |
| D10 | Daily Demand Forecasting Orders |
| D12 | Beijing PM2.5 |
| D14 | Concrete Compressive Strength |
| D16 | Dow Jones Index |
| D17 | Energy Efficiency |
| D22 | Istanbul Stock Exchange |
| D23 | Bike Sharing |
| D30 | Household Power Consumption |
| D31 | Online News Popularity |

### 🎯 Classification — 16 Datasets

| ID | Dataset |
| :---: | :--- |
| D02 | User Knowledge Modeling |
| D04 | Wireless Indoor Localization |
| D05 | Car Evaluation |
| D06 | Fertility |
| D07 | Qualitative Bankruptcy |
| D09 | Heart Disease |
| D11 | Blood Transfusion |
| D13 | Echocardiogram |
| D18 | Glass Identification |
| D19 | Hepatitis |
| D24 | Occupancy Detection |
| D25 | Census Income |
| D26 | Coronavirus |
| D27 | Autism Screening Adult |
| D28 | Default of Credit Card Clients |
| D29 | Banknote Authentication |

### 🧩 Clustering — 3 Datasets

| ID | Dataset |
| :---: | :--- |
| D15 | Liver Disorders |
| D20 | Wholesale Customers |
| D21 | Travel Reviews |

---

## ⚙️ Application Modules

### 🏠 Home

Provides an introduction to the application, its capabilities,
and access to the main Machine Learning modules.

### 📁 Datasets

An organized dataset catalog that provides information about
dataset characteristics, features, target variables,
Machine Learning tasks, and recommended models.

### 🧠 Algorithms

Introduces the supported Machine Learning algorithms and
their fundamental concepts.

### 🔮 Predictor

An interactive prediction environment that allows users to:

- Enter custom feature values
- Select available Machine Learning models
- Generate predictions using pre-trained models
- Explore prediction results
- View recommended models and performance information

### 📋 Evaluation

A dedicated environment for comparing Machine Learning models
using task-specific performance metrics.

### 📊 Visualization

An analytical environment for exploring relationships,
patterns, feature contributions, and model performance
through visual representations.

### 🎨 Settings

Provides appearance customization through light and dark
interface modes.

---

## 📐 Model Evaluation

### Regression Metrics

| Metric | Description |
| :--- | :--- |
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| R² Percentage | R² expressed as a percentage |
| Accuracy | Application-specific RMSE-based performance score |

> Regression Accuracy in Machine Learning Studio is a custom
> RMSE-based performance score, not conventional classification
> accuracy. Its normalization may vary between datasets.

### Classification Metrics

| Metric | Description |
| :--- | :--- |
| Accuracy | Percentage of correctly classified observations |
| F1 Score | Weighted balance between precision and recall |
| Macro F1 | Unweighted average of class-level F1 scores |
| Error Rate | Percentage of incorrectly classified observations |

### Clustering Evaluation

Clustering results are analyzed through cluster distributions,
visual representations, and available clustering quality metrics.

### 🏆 Recommended Model Selection

The application identifies recommended models based on
task-specific evaluation criteria:

- **Regression:** Lowest RMSE
- **Classification:** Highest Accuracy
- **Clustering:** Available clustering quality indicators

Model performance depends on the dataset, preprocessing,
evaluation procedure, and selected algorithm.

---

## 📉 Data Visualization

Machine Learning Studio provides a collection of analytical
visualizations, depending on the selected dataset and task.

### Regression Visualizations

- Feature importance and feature contribution
- Correlation heatmaps
- Actual vs. predicted values
- Model performance comparison
- Residual analysis

### Classification Visualizations

- Class distribution
- Feature distribution by class
- PCA projection
- Correlation heatmaps
- Feature importance
- Classification model comparison

### Clustering Visualizations

- Cluster distribution
- Cluster visualization
- Dataset-specific clustering analysis

These visualizations help users understand dataset
characteristics, model behavior, and analytical results.

---

## 🛠️ Technologies & Libraries

| Category | Technologies |
| :--- | :--- |
| Programming Language | Python |
| Graphical User Interface | Tkinter, CustomTkinter |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Scientific Computing | SciPy |
| Data Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib, Cloudpickle |
| Image Processing | Pillow |
| Spreadsheet Processing | OpenPyXL, XLRD, Python-Calamine |
| Application Packaging | PyInstaller |
| Version Control | Git, Git LFS |
| Repository Hosting | GitHub |

---

## 📦 Repository Structure

```text
Machine-Learning-Studio/
│
├── assets/
│   ├── DataSets/
│   ├── icons/
│   ├── page_images/
│   ├── MLStudio.ico
│   └── MLStudio.png
│
├── Classification/
│   ├── DataSets/
│   │   └── Processed/
│   ├── Metadata/
│   └── ModelsOutcome/
│
├── Clustering/
│   ├── Datasets/
│   │   └── Processed/
│   ├── Metadata/
│   └── ModelsOutcome/
│
├── Regression/
│   ├── DataSets/
│   │   └── Processed/
│   ├── Metadata/
│   └── ModelsOutcome/
│
├── Release/
│   └── Machine Learning Studio/
│       ├── Machine Learning Studio.exe
│       └── _internal/
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

### Directory Description

| Directory | Purpose |
| :--- | :--- |
| assets | Application icons, images, and graphical resources |
| DataSets / Datasets | Original and processed datasets |
| Metadata | Dataset information and model evaluation metadata |
| ModelsOutcome | Serialized pre-trained Machine Learning models |
| Release | Complete standalone Windows application |
| _internal | Runtime dependencies and application resources |

The current repository distribution focuses on the executable
application, datasets, trained models, metadata, and graphical
resources rather than development source files.

---

## 🚀 Installation & Usage

Machine Learning Studio is distributed as a standalone Windows
desktop application.

No separate Python installation, virtual environment, or
manual dependency installation is required to run the
packaged application.

### 1. Install Git LFS

Git LFS is required to retrieve large datasets, trained models,
and executable dependencies.

Download Git LFS from:

https://git-lfs.com/

After installation, open PowerShell and run:

```powershell
git lfs install
```

### 2. Clone the Repository

```powershell
git clone https://github.com/sara-mollaie/Machine-Learning-Studio.git
```

### 3. Navigate to the Project Directory

```powershell
cd Machine-Learning-Studio
```

### 4. Download Large Files

```powershell
git lfs pull
```

### 5. Launch the Application

```powershell
& ".\Release\Machine Learning Studio\Machine Learning Studio.exe"
```

Alternatively, navigate to:

```text
Release/Machine Learning Studio/
```

Double-click:

```text
Machine Learning Studio.exe
```

> [!IMPORTANT]
> Keep the executable and the `_internal` directory together.
> Moving or deleting the internal runtime files may prevent
> the application from launching correctly.

> [!NOTE]
> The repository contains large Machine Learning models and
> application dependencies. Downloading the complete repository
> may require several gigabytes of disk space.

---

## 💻 System Requirements

- Windows operating system compatible with the packaged build
- Sufficient available disk space
- Sufficient RAM for loading Machine Learning models
- Git and Git LFS for cloning the repository

Python and its required runtime libraries are included in
the packaged executable distribution.

The `requirements.txt` file documents the Python dependencies
used during application development.

---

## 🖼️ Application Screenshots

Application screenshots will showcase the following modules:

- Home — Light and Dark Mode
- Dataset Explorer
- Machine Learning Algorithms
- Regression Predictor
- Classification Predictor
- Clustering Analysis
- Model Evaluation
- Data Visualization

---

## 📚 Educational Purpose

Machine Learning Studio is developed as an educational and
practical platform for students, researchers, and individuals
interested in Data Science and Machine Learning.

The application demonstrates how different Machine Learning
techniques can be applied to real-world datasets and how
their results can be evaluated, compared, and interpreted.

Predictions and analytical results are intended for educational
and exploratory purposes and should not be treated as
professional medical, financial, or other high-stakes advice.

---

## 🔮 Future Enhancements

Potential improvements for future versions include:

- Custom dataset import
- Additional Machine Learning algorithms
- User-defined model training
- Advanced hyperparameter optimization
- Automated preprocessing workflows
- Exportable analytical reports
- Expanded model explainability
- Additional platform support

---

## 👩‍💻 Developer

**Sara Mollaie**

Machine Learning Studio — Developer

[![GitHub](https://img.shields.io/badge/GitHub-sara--mollaie-181717?style=for-the-badge&logo=github)](https://github.com/sara-mollaie)

---

<div align="center">

### Machine Learning Studio

**Exploring Data. Understanding Models. Discovering Insights.**

Developed by **Sara Mollaie**

⭐ If you find this project useful, consider giving it a star.

</div>
