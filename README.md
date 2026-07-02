# Overreaction as an Indicator for Momentum in Algorithmic Trading: A Case of AAPL Stocks

## Overview

This repository contains the Python/PyTorch implementation of the research paper "[Overreaction as an indicator for momentum in algorithmic trading: A Case of AAPL stocks](https://arxiv.org/pdf/2602.18912v1)" by Szymon Lis, Robert Ślepaczuk, and Paweł Sakowski. The paper explores the use of machine learning models to predict and monetize short-term market overreactions as momentum signals in algorithmic trading. Using high-frequency emotional data derived from Twitter messages and volatility-normalized returns, the study focuses on Apple Inc. (AAPL) stock and evaluates the predictive power of various nonlinear classifiers across multiple intraday frequencies.

This repository provides the implementation of the methodologies described in the paper, including data preprocessing, feature extraction, model training, and evaluation of trading strategies informed by the model predictions.

---

## Core Concept

The paper investigates **short-term market overreactions**, defined as extreme stock return realizations relative to contemporaneous volatility and transaction costs. By leveraging **transformer-based emotional features** extracted from Twitter messages and combining them with volatility-normalized returns, the authors aim to predict these overreactions.

This prediction problem is framed as a **three-class classification task**:
1. **Positive Overreaction**
2. **Neutral Movement**
3. **Negative Overreaction**

Various machine learning models, such as XGBoost, Random Forests, Deep Neural Networks, and Bidirectional LSTMs, are trained to predict these classes at different intraday frequencies (1, 5, 10, and 15 minutes). The predictions are then translated into trading strategies, which are evaluated using risk-adjusted performance measures and statistical tests.

### Key Findings
1. Machine learning models significantly outperform traditional benchmark rules for predicting overreactions at ultra-short horizons.
2. Behavioral momentum effects dominate at intermediate frequencies (e.g., 10-minute intervals).
3. Explainability analysis using SHAP (SHapley Additive exPlanations) highlights the importance of volatility and negative emotions (e.g., fear and sadness) in driving overreaction predictions.

---

## Repository Structure

```
├── data/
│   ├── raw/                # Raw datasets (e.g., Twitter emotion features, AAPL intraday stock data)
│   └── processed/          # Preprocessed data ready for modeling
├── notebooks/
│   ├── data_preprocessing.ipynb    # Notebook for data cleaning and feature engineering
│   ├── exploratory_analysis.ipynb  # Exploratory data analysis (EDA) and visualization
│   └── model_training.ipynb        # Training machine learning models and evaluating predictions
├── models/
│   ├── xgboost_model.pkl           # Pretrained XGBoost model
│   ├── lstm_model.pth              # Pretrained Bidirectional LSTM model
│   └── random_forest_model.pkl     # Pretrained Random Forest model
├── scripts/
│   ├── train.py                    # Script for training models
│   ├── evaluate.py                 # Script for evaluating model performance and trading strategies
│   └── shap_analysis.py            # Script for SHAP-based explainability analysis
├── requirements.txt                # Dependencies and required Python packages
└── README.md                       # Project documentation
```

---

## Features of the Code

### 1. **Data Processing**
The code includes functionality to:
- **Normalize volatility and returns**: Calculate metrics for identifying overreactions in AAPL stock data.
- **Extract emotional features**: Use transformer-based models to analyze Twitter messages and assign sentiment scores (e.g., fear, sadness, joy).
- **Combine datasets**: Merge stock market data and Twitter-derived sentiment features into a comprehensive intraday dataset.

### 2. **Model Training**
The repository supports multiple machine learning models for predicting overreactions:
- **XGBoost**
- **Random Forests**
- **Deep Neural Networks**
- **Bidirectional LSTMs**

Each model is trained on labeled data to predict the likelihood of positive, neutral, or negative overreactions.

### 3. **Trading Strategy Evaluation**
Predictions from the models are converted into actionable trading signals:
- **Buy**: If positive overreaction is predicted.
- **Sell**: If negative overreaction is predicted.
- **Hold**: If neutral movement is predicted.

The repository includes scripts to backtest these strategies and evaluate their performance using metrics such as:
- **Sharpe Ratio**
- **Sortino Ratio**
- **Profit and Loss (PnL)**

### 4. **Explainability Analysis**
Using SHAP, the repository provides tools to:
- Analyze feature importance in driving model predictions.
- Understand the contribution of emotional sentiment (e.g., fear, sadness) and volatility to predicted overreactions.

---

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/aapl-overreaction-momentum.git
   cd aapl-overreaction-momentum
   ```

2. Set up a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Usage

### Data Preprocessing
Run the `data_preprocessing.ipynb` notebook to clean and preprocess the raw datasets. This step generates the `processed` data, which is used for training models.

### Training Models
Use the `train.py` script to train any of the supported models:
```
python scripts/train.py --model xgboost
```

### Evaluating Trading Strategies
Evaluate the performance of the trading strategies using:
```
python scripts/evaluate.py --model xgboost
```

### Explainability Analysis
Visualize feature importance and interpret the predictions using SHAP:
```
python scripts/shap_analysis.py --model xgboost
```

---

## Results

Key results from the paper:
- **Outperformance of ML models**: At ultra-short horizons (1-5 minutes), machine learning models consistently outperformed traditional rules in predicting overreactions.
- **Behavioral momentum effects**: At intermediate frequencies (e.g., 10 minutes), classical momentum patterns linked to behavioral finance dominated.
- **Feature insights**: Volatility and negative emotional sentiment (fear, sadness) were identified as the most important features driving predictions.

---

## Dependencies

- Python 3.8+
- PyTorch
- scikit-learn
- XGBoost
- pandas
- numpy
- matplotlib
- SHAP
- transformers (for sentiment analysis)

Refer to `requirements.txt` for a complete list of dependencies.

---

## Contribution

Contributions to this repository are welcome. If you find bugs or want to suggest improvements, feel free to open an issue or submit a pull request.

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.