# 🧠 **Machine Learning (ML) Cheat Sheet for Data Scientists (Fresher to Intermediate Level)** 
— covering algorithms, formulas, Python syntax, and key concepts you'll use in practice and interviews.

---

## 📘 1. **Types of Machine Learning**

| Type | Description | Examples |
|------|-------------|----------|
| **Supervised Learning** | Labeled data, predict output | Regression, Classification |
| **Unsupervised Learning** | Unlabeled data, find patterns | Clustering, Dimensionality Reduction |
| **Semi-Supervised** | Few labeled, mostly unlabeled | Text classification |
| **Reinforcement Learning** | Agent learns from environment | Game AI, Robotics |

---

## 🧮 2. **Common Algorithms**

### 🔹 Supervised Learning

| Task | Algorithm | Use Case |
|------|-----------|----------|
| Regression | Linear Regression | Predict price, sales |
| Classification | Logistic Regression | Yes/No, Fraud detection |
| Classification | Decision Tree | Customer churn |
| Classification | Random Forest | Ensemble learning |
| Classification | SVM | Image recognition |
| Regression + Classification | XGBoost / LightGBM | Tabular data (Kaggle, real-world) |
| Classification | KNN | Pattern recognition |

### 🔹 Unsupervised Learning

| Task | Algorithm | Use Case |
|------|-----------|----------|
| Clustering | K-Means | Customer segmentation |
| Clustering | Hierarchical | Gene analysis |
| Dim. Reduction | PCA | Feature compression |
| Association | Apriori / FP-Growth | Market basket analysis |

---

## 📏 3. **Evaluation Metrics**

### Classification:
| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | \( \frac{TP + TN}{Total} \) | General performance |
| Precision | \( \frac{TP}{TP + FP} \) | False positives matter |
| Recall | \( \frac{TP}{TP + FN} \) | False negatives matter |
| F1 Score | \( 2 \cdot \frac{P \cdot R}{P + R} \) | Balance P & R |
| ROC-AUC | Curve Score | Class separation power |

### Regression:
| Metric | Formula | Use Case |
|--------|---------|----------|
| MAE | \( \frac{1}{n} \sum |y - \hat{y}| \) | Average error |
| MSE | \( \frac{1}{n} \sum (y - \hat{y})^2 \) | Penalizes large errors |
| RMSE | \( \sqrt{MSE} \) | More interpretable |
| R² Score | \( 1 - \frac{SS_{res}}{SS_{tot}} \) | Model fit quality |

---

## 📊 4. **Data Preprocessing Techniques**

| Task | Method | Tool |
|------|--------|------|
| Missing Values | `fillna()`, `SimpleImputer` | pandas, sklearn |
| Scaling | `StandardScaler`, `MinMaxScaler` | sklearn |
| Encoding | `LabelEncoder`, `OneHotEncoder` | sklearn |
| Outlier Removal | IQR, Z-score | NumPy, pandas |
| Feature Selection | `SelectKBest`, `RFE`, `PCA` | sklearn |

---

## ⚙️ 5. **Model Building (Scikit-Learn Syntax)**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
```

---

## 🧪 6. **Model Validation Techniques**

| Method | Description |
|--------|-------------|
| Train-Test Split | Simple and fast |
| K-Fold Cross-Validation | Better generalization |
| Stratified K-Fold | Balanced class proportions |
| GridSearchCV / RandomizedSearchCV | Hyperparameter tuning |

---

## 🧠 7. **Deep Learning (Basics)**

| Term | Meaning |
|------|--------|
| Epoch | One full pass over data |
| Batch Size | Data processed before updating weights |
| Optimizers | SGD, Adam |
| Loss Function | MSE (regression), CrossEntropy (classification) |
| Frameworks | TensorFlow, Keras, PyTorch |

---

## 🧰 8. **Popular Python Libraries**

| Purpose | Library |
|--------|---------|
| ML | Scikit-learn |
| Deep Learning | TensorFlow, PyTorch |
| EDA | pandas, seaborn, matplotlib |
| Data Handling | NumPy |
| NLP | NLTK, spaCy |
| Model Tuning | Optuna, Hyperopt |

---

## 📂 9. **Real-World Use Cases**

| Domain | Problem | ML Task |
|--------|--------|--------|
| Retail | Sales prediction | Regression |
| Finance | Credit scoring | Classification |
| Healthcare | Disease diagnosis | Classification |
| E-commerce | Customer clustering | Clustering |
| Marketing | Campaign optimization | A/B Testing, Classification |

---

## 🧠 10. **Bonus: Pipeline Creation (sklearn)**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipe.fit(X_train, y_train)
```

