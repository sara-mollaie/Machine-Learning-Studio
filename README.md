# Machine Learning Studio

A Python-based desktop application designed to explore datasets and apply Machine Learning algorithms through an interactive graphical interface.

## Project Overview

Machine Learning Studio is an educational and practical application developed to provide an accessible environment for working with Machine Learning algorithms, exploring datasets, and analyzing results.

The application is designed to support multiple Machine Learning approaches within a unified desktop interface.

## Features

- Interactive desktop interface
- Dataset exploration and management
- Machine Learning model execution
- Model output and result management
- Organized project architecture

## Machine Learning Algorithms

### 1. Clustering
**Status: Implemented**

- K-Means Clustering

### 2. Classification
**Status: Planned**

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### 3. Regression
**Status: Planned**

- Linear Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Technologies Used

### Programming Language
- Python

### User Interface
- CustomTkinter

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- SciPy
- Joblib
- Cloudpickle

### Data Visualization
- Matplotlib
- Seaborn

### Image Processing
- Pillow (PIL)

*The technology list will be updated according to the actual project dependencies.*

## Project Structure

```text
Machine Learning Studio/
│
├── Clustering/
│   ├── Datasets/
│   ├── Metadata/
│   ├── ModelsOutcome/
│   ├── Liver Disorders.py
│   ├── Travel Reviews.py
│   └── Wholesale customers.py
│
├── UserInterfaceLayer/
│   ├── Datasets.py
│   ├── Dialogs.py
│   ├── HomeModule.py
│   ├── MainView.py
│   ├── Theme.py
│   └── Window.py
│
├── assets/
├── .gitignore
├── Main.py
├── README.md
├── requirements.txt
└── settings.json
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sara-mollaie/Machine-Learning-Studio.git
```

### 2. Navigate to the project directory

```bash
cd Machine-Learning-Studio
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the main application:

```bash
python Main.py
```

## Screenshots

Screenshots of the application interface and its Machine Learning modules will be added here.

## Future Improvements

- Implement Classification algorithms
- Implement Regression algorithms
- Expand dataset support
- Improve model evaluation and visualization
- Enhance the user interface

## Author

**Sara Mollaie**

GitHub: [@sara-mollaie](https://github.com/sara-mollaie)

---

*Machine Learning Studio – Exploring Data, Building Models, Learning from Results.*