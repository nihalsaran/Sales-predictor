# Decathlon Sales Prediction

## Table of Contents

- [Decathlon Sales Prediction](#decathlon-sales-prediction)
  - [About](#about)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Data](#data)
    - [Training the Model](#training-the-model)
    - [Evaluation](#evaluation)
    - [Generating Graphs](#generating-graphs)
  - [Contributing](#contributing)
  - [License](#license)

## About

This project is focused on predicting sales for the company Decathlon using a Machine Learning model. Additionally, it generates four key graphs to visualize the data and model predictions.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python (>=3.6)
- Required Python libraries (Pandas, NumPy, Matplotlib, Scikit-Learn)

## Getting Started

### Installation

To set up the project environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/nihalsaran/Sales-predictor.git
   cd Sales-prediction
   ```

2. Install the required Python libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

## Usage

### Data

Replace the sample data in `your_dataset.csv` with your actual sales data. Ensure that the CSV file contains columns: 'Year', 'Turnover', 'TV', and 'Newspaper'.

### Training the Model

Run the following command to train the Machine Learning model:

```bash
python train_model.py
```

### Evaluation

After training, the model's performance will be evaluated, and the Mean Squared Error (MSE) and R-squared (R2) score will be displayed.

### Generating Graphs

Run the following command to generate the four requested graphs:

```bash
python generate_graphs.py
```

The graphs include:
1. Turnover vs. Years
2. TV vs. Advertising Cost
3. Newspaper vs. Advertising Cost



