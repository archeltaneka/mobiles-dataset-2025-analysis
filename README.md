# 📱Mobiles Dataset Analysis

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://archeltaneka-mobiles-dataset-2025-analysis-app-fz8tnw.streamlit.app/)
![Tests](https://github.com/archeltaneka/mobiles-dataset-2025-analysis/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/github/archeltaneka/mobiles-dataset-2025-analysis/graph/badge.svg?token=O061FNP8I4)](https://codecov.io/github/archeltaneka/mobiles-dataset-2025-analysis)

A fully interactive data exploration dashboard that transforms the messy 2025 smartphone dataset into clean insights, pricing intelligence, and trend visualizations.

Built with Streamlit, Plotly, and a custom data-processing pipeline.

🔗 Live App: https://archeltaneka-mobiles-dataset-2025-analysis-app-fz8tnw.streamlit.app/

📂 Dataset: [Kaggle – Mobiles Dataset 2025](https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025)

## 🚀 Highlights

- Cleaned and processed 100+ smartphone attributes using a custom wrangling pipeline.
- Created an interactive dashboard to explore market trends, pricing, and feature comparisons.
- Implemented correlation analysis to understand which specs affect price.
- Added automated tests + GitHub Actions CI + coverage reporting.
- Fully deployed to Streamlit Cloud.

## 🖥️ Key Features

### 🔍 Smart Filters

Explore the dataset by:
- Manufacturer / brand
- Budget tier
- Launch year
- Price range
- Regional price (USD, INR, AED, PKR, CNY)

### 📊 Visual Insights

Includes:
- Price distribution by brand
- Feature comparisons
- Radar & polar correlation charts
- Trend analysis across years
- Market segmentation visualizations

### 🧠 Data Processing

- Automated cleaning of camera resolution, weights, memory formats, etc.
- Feature normalization and type consistency checks
- Consistent schema transformations

### 🧪 Testing & CI

- Unit tests for all data-processing functions
- Pytest + Coverage
- GitHub Actions automated workflow
- Codecov integration

## 🛠️Tech Stack

- Python
- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-learn

## 📃Requirements
- Python 3.10+

## 📦Installation

```
git clone https://github.com/archeltaneka/mobiles-dataset-2025-analysis
cd mobiles-dataset-2025-analysis
pip install -r requirements.txt
streamlit run app.py
```

## 🗂 Project Structure

```
mobiles-dataset-2025-analysis/
├── README.md
├── app.py                              # streamlit app
├── data    
│   ├── Mobiles Dataset (2025).csv      # raw dataset
├── requirements.txt
├── src
│   ├── analytics
│   │   ├── __init__.py
|   |   ├── clustering.py               # phone market segmentation
|   |   ├── scoring.py                  # phone value-for-money scoring
│   ├── cleaning
│   │   ├── __init__.py
│   │   ├── feature_extraction.py       # feature extraction
│   │   ├── pipeline.py                 # data cleaning pipeline
│   │   ├── preprocessing.py            # data preprocessing
│   ├── data
│   │   ├── __init__.py
│   │   └── mobiles.py                  # data loader
│   ├── __init__.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_clustering.py
    ├── test_data_loader.py
    ├── test_feature_extraction.py
    ├── test_pipeline.py
    ├── test_preprocessing.py
    ├── test_scoring.py
```

## 🍿Demo Video

https://github.com/user-attachments/assets/da285d5b-0ec4-4b59-bc97-0c635f4f152c

## 📄 License

MIT License © 2025 Archel Taneka

## ⚙️ Want to contribute?

PRs, suggestions, and issues are welcome.


