# 🎩 Hogwarts Logistic Regression

## 📚 Overview

This project consists of building a multi-class logistic regression classifier (One-vs-All) from scratch in order to recreate the famous Sorting Hat.

The project is divided into three main parts:

1. Data Analysis

2. Data Visualization

3. Logistic Regression (Machine Learning)

No high-level ML libraries are used for the regression itself.
The objective is to deeply understand data processing, feature selection, and gradient descent optimization.


-------------------------------------------------------------------------------------------------------------------------------------------------
## 🛠 Installation

### 🐍 1️⃣ Create a Python Virtual Environment
It is strongly recommended to use a virtual environment to isolate project dependencies.

Create the virtual environment:
```sh
python3 -m venv .venv
```

Activate the virtual environment:

Linux / macOS
```sh
source .venv/bin/activate
```

Windows
```sh
.venv\Scripts\activate
```


### 📥 2️⃣ install Dependencies

The project includes a requirements.txt file containing all required Python packages.
```sh
pip install -r requirements.txt
```


### 🗂 3️⃣ Extract the Dataset
The datasets are provided inside a compressed archive:

Manual extraction
```sh
tar -xvf datasets.tgz.xz;tar -xvf datasets.tgz
```
This will extract:

dataset_train.csv et dataset_test.csv


-------------------------------------------------------------------------------------------------------------------------------------------------
## 🧪 Part I — Data Analysis
Before building any model, it is essential to understand the dataset.

We begin by analyzing the feature of the dataset. 
So we exclude the feature like First Name, Birthday. 
And use some metrics to analyse the pertinent feature. 

🔎 describe.py
 ```sh
describe.py dataset/dataset_train.csv
```
<img width="925" height="318" alt="describefeatureDSLR" src="https://github.com/user-attachments/assets/85e153f3-a153-4e28-a4e6-64efadc073c0" />

This step allows us to:

- Detect abnormal ranges
- Identify possible outliers
- Understand feature scaling needs


-------------------------------------------------------------------------------------------------------------------------------------------------
## 📊 Part II — Data Visualization

Visualization helps us select meaningful features for classification.

### 📈 histogram.py
Question:
Which Hogwarts course has a homogeneous score distribution between all four houses?

This script plots histograms grouped by house.
<img width="1301" height="614" alt="histogramDSLR" src="https://github.com/user-attachments/assets/112f174d-f998-44a1-85e0-b419405eab4c" />


Goal:
- Identify features with similar distributions across classes.
- Detect potential weak discriminators.

### 🔵 scatter_plot.py
Question:
What are the two features that are similar?

Displays a scatter plot between two selected features.
<img width="1301" heigh<img width="992" height="795" alt="scatter_plotDSLR" src="https://github.com/user-attachments/assets/6a960cef-96e5-42dd-85b5-3b4b772f8818" />

Goal:
- Identify correlated features.
- Detect redundancy
- Understand linear relationships.

### 📌 pair_plot.py
Question:
From this visualization, which features are you going to use for your logistic regression?

Generates a full scatter matrix of all numerical features.
<img width="2864" height="1504" alt="pair_plotDSLR" src="https://github.com/user-attachments/assets/397d6449-425d-42f4-b582-0f4f7907094e" />

Goal:
- Detect class separability
- Identify linear boundaries
- Select the most relevant features
- This is a critical step before training.


-------------------------------------------------------------------------------------------------------------------------------------------------
## 🧠 Part III — Logistic Regression (Magic Hat)
We implement a multi-class logistic regression using One-vs-All (One-vs-Rest) strategy.

Each house is treated as a separate binary classifier.

### 🏗 logreg_train.py

#### Responsibilities:
- Feature selection
- Feature normalization
- Initialize weights
- Apply Gradient Descent
- Minimize logistic loss
- Save trained weights to a file

-------------------------------------------------------------------------------------------------------------------------------------------------
#### 🔁 Gradient Descent Variants
To train the logistic regression model, we must minimize the cost function using a gradient-based optimization algorithm. Several variants of gradient descent can be used:


-------------------------------------------------------------------------------------------------------------------------------------------------
#### 1️⃣ Batch Gradient Descent
Batch Gradient Descent computes the gradient using the entire training dataset at each iteration:
Where the gradient is calculated over all m training samples.

usage:
```sh
python3 logreg_train.py datasets/dataset_train.csv
```

Advantages:
- Stable convergence
- Smooth and predictable updates
- Deterministic behavior

Disadvantages:
- Slow when the dataset is large
- Computationally expensive per iteration

This is the most common approach in academic implementations and is typically required for this project.


-------------------------------------------------------------------------------------------------------------------------------------------------
#### 2️⃣ Mini-Batch Gradient Descent

Mini-Batch Gradient Descent is a compromise between the two previous methods.
It updates parameters using a small batch of samples (e.g., 32, 64, 128 samples):

use:
```sh
python3 logreg_train.py datasets/dataset_train.csv  -a mbGD
```

Advantages:
- Faster than full batch
- More stable than SGD
- Efficient matrix operations
- Widely used in practical machine learning

Disadvantages:
- less stable than full batch (minime)


-------------------------------------------------------------------------------------------------------------------------------------------------
#### 🔁 Mini-Batch Stochastic Gradient Descent

To train the logistic regression model, we minimize the cost function using a variant of Stochastic Gradient Descent with mini-batches.
Instead of computing the gradient on the entire dataset (Batch Gradient Descent),
 or updating the weights after every single sample (pure Stochastic Gradient Descent),
 we split the dataset into small subsets called mini-batches.
 
At each iteration, the model parameters are updated using only one mini-batch:
- The gradient is computed over a subset of samples
- The dataset is shuffled between epochs to improve learning

use:
```sh
python3 logreg_train.py datasets/dataset_train.csv  -a SGD
```

Advantage:
- This approach combines the advantages of both Batch Gradient Descent and Stochastic Gradient Descent:
- 🚀 Faster convergence than full batch
- 📉 Less noisy than pure SGD
- ⚡ More efficient matrix operations
- 🔄 Better generalization due to stochasticity
- 🧠 Scales better to larger datasets


-------------------------------------------------------------------------------------------------------------------------------------------------
### 🔮 logreg_predict.py
Usage
```sh
logreg_predict.py dataset_test.csv weights.txt
```

#### Responsibilities
- Load trained weights
- Apply normalization
- Compute probabilities for each class
- Select class with highest probability
- Generate output csv file white the prediction

This model has an accuracy score of 99%.

### Contributor
- [vetuedenoir](https://github.com/vetuedenoir)
- [mouxprime](https://github.com/mouxprime)




