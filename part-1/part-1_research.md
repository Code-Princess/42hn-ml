# Part 1: Machine Learning - Introduction

Welcome to the first part of the **Artificial Intelligence and Machine Learning** module.  
In this section, you'll begin your journey into machine learning and train your first models.

Don’t just follow the subject blindly—strive to **understand** the tools you're using.  
Read the documentation, and be prepared to explain:

- What each step does
- Why it’s important
- The advantages and limitations of each tool

---

## What is Machine Learning?

Machine learning is a branch of computer science focused on building systems that **learn from data**.  
Rather than following fixed rules like traditional programs, ML models uncover hidden patterns in datasets to make predictions or decisions.

### Research Topics

Before getting started do your research on machine learning and be able to **at least** answer the following questions:

- What is the difference between supervised and unsupervised learning?
# Supervised vs. Unsupervised Learning

| Feature                     | **Supervised Learning**                             | **Unsupervised Learning**                          |
|----------------------------|-----------------------------------------------------|----------------------------------------------------|
| **Labeled Data**           | Requires labeled data (input-output pairs)          | No labeled data required                           |
| **Objective**              | Predict output from input                           | Discover hidden patterns or structure              |
| **Typical Tasks**          | Classification, Regression                          | Clustering, Dimensionality Reduction               |
| **Example Algorithms**     | Linear Regression, Decision Trees, SVM, Neural Nets | K-Means, Hierarchical Clustering, PCA, t-SNE       |
| **Output**                 | Predict specific labels or continuous values        | Groupings, clusters, or compressed representation  |
| **Example Use Case**       | Email spam detection, disease diagnosis             | Customer segmentation, anomaly detection           |
| **Data Efficiency**        | Needs large labeled datasets                        | Can work with raw, unlabeled data                  |

- What are types of supervised learning?

   Classification and Regression

- What’s the difference between **classification** and **regression**? What output does each produce?
## 📊 Classification vs. Regression

| Feature              | **Classification**                                    | **Regression**                                           |
|----------------------|--------------------------------------------------------|-----------------------------------------------------------|
| **Output Type**      | Discrete categories or class labels                    | Continuous numeric values                                 |
| **Goal**             | Assign input to a specific class                       | Predict a quantity based on input                         |
| **Examples**         | Email: Spam or Not Spam<br>Image: Cat, Dog, etc.       | Predict house prices<br>Forecast temperature              |
| **Typical Algorithms** | Logistic Regression, SVM, Decision Trees, k-NN       | Linear Regression, SVR, Random Forest Regressor           |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score                 | Mean Squared Error (MSE), Mean Absolute Error (MAE), R²   |

- How do you determine whether a problem is classification or regression?
## 🧭 How to Tell: Classification or Regression?

| Question                                          | Classification | Regression |
|--------------------------------------------------|----------------|------------|
| Is the output a **category or class label**?     | Yes            | No         |
| Is the output a **real/continuous number**?      | No             | Yes        |
| Is the number of possible outputs **finite**?    | Yes            | No         |
| Can the outputs be **ranked on a numeric scale**?| No             | Yes        |
| Are you answering **“What type?”**               | Yes            | No         |
| Are you answering **“How much?” or “How many?”** | No             | Yes        |

- Are there ML problems that fall outside classification and regression?
## 🧠 ML Problem Types Beyond Classification & Regression

| Problem Type              | Description                                                                 | Example Use Cases                          |
|---------------------------|-----------------------------------------------------------------------------|--------------------------------------------|
| **Clustering**            | Grouping similar data points without labels                                 | Customer segmentation, topic modeling      |
| **Dimensionality Reduction** | Reducing the number of features while preserving structure                  | Data visualization (e.g., PCA), noise reduction |
| **Anomaly Detection**     | Identifying rare or unusual data points                                     | Fraud detection, network intrusion         |
| **Recommendation Systems**| Predicting user preferences based on past behavior                          | Movie or product recommendations           |
| **Reinforcement Learning**| Learning via reward signals through trial and error                         | Robotics, game playing (e.g., AlphaGo)     |
| **Self-supervised Learning** | Using the data itself to generate labels                                  | Pretraining models like BERT, SimCLR       |
| **Multi-task Learning**   | Solving multiple learning tasks simultaneously                              | Joint classification and regression        |

- What is skewed data and how to mitigate it's effect?
   
   📉 What Is Skewed Data?
   Skewed data occurs when the distribution of your target variable (typically in classification) is imbalanced—i.e., some classes appear far more frequently than others.
   
   Example: In a medical dataset, 95% of patients might be healthy, and only 5% have a rare disease.
   
   Result: A model could "cheat" by always predicting the majority class and still achieve high accuracy.
   
## ⚖️ Techniques to Handle Skewed Data

| Technique                       | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| **Resampling: Oversampling**    | Duplicate examples from the minority class (e.g. SMOTE)                     |
| **Resampling: Undersampling**   | Remove examples from the majority class                                    |
| **Class Weights**               | Assign higher weights to minority class during training                    |
| **Anomaly Detection Approach**  | Treat minority class as an outlier or rare event                           |
| **Change Evaluation Metrics**   | Use metrics like Precision, Recall, F1-score, AUC instead of Accuracy       |
| **Generate Synthetic Data**     | Use techniques like GANs or SMOTE to synthesize new minority examples       |
| **Stratified Sampling**         | Ensure each class is proportionally represented in training/test splits     |


---

## Predicting Bike Sharing Demand

Explore the dataset to understand distributions, correlations, and feature relationships.  
Use appropriate plots and explain your choices—why you chose that graph, and what alternatives exist.

### Prepare:

1. Gain insights from the data through exploration.

    bikeSharing_plots.ipynb
2. Clean and adjust the data as necessary.

    bikeSharing_dataCheck.ipynb, bikeSharing_anomalies.ipynb, bikeSharing_dataClean.ipynb
3. Build a **preprocessing pipeline** using `scikit-learn`.
   - Wrap the pipeline creation into a **reusable function** that accepts any estimator.

### Model Training

Train models to predict hourly bike rentals using:

- `LinearRegression`
- `RandomForestRegressor`
- `XGBRegressor`
- `GridSearchCV` for hyperparameter tuning

Evaluate your models using **multiple performance metrics**.

### Dataset

Download the dataset from the UCI Machine Learning Repository:

👉 [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

---

## Requirements

You are required to use the following tools:

- `uv` – Dependency management
- `jupyter` – Interactive notebook environment
- `pandas` – Data manipulation
- `seaborn` – Visualization
- `scikit-learn` – ML toolkit
- `mlflow` – Experiment tracking and logging
  - in terminal type:
  
    (.venv) level3@admins-Mac-mini 42hn-ml % mlflow server --host localhost --port 5000 
