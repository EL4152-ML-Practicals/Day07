# 📊 Day 07 : Regression Analysis

A comprehensive guide to three regression implementations using Python, scikit-learn, and scipy.

---

## 📚 Three Implementation Files

### 1️⃣ **Regression.ipynb** - Simple Linear Regression (scipy)

#### 🎯 What it does:

Basic linear regression using scipy's `stats.linregress()` - **No train/test split**

#### 🔑 Key Steps:

```
📥 Input: x and y arrays (hardcoded values)
     ↓
📈 Calculate: slope, intercept using scipy.stats
     ↓
🎨 Plot: Scatter plot + line of best fit
     ↓
📊 Output: Correlation coefficient (r value)
```

#### 💡 Quick Memory:

- **Quick & Simple** ✨
- Uses scipy, not sklearn
- Perfect for understanding basic concept
- Shows regression line visually

#### 🔢 Key Variables:

- `x`, `y` → Input data
- `slope`, `intercept`, `r` → Regression parameters
- `mymodel` → Predicted y values

---

### 2️⃣ **RegressionEX.ipynb** - Simple Linear Regression (sklearn)

#### 🎯 What it does:

Linear regression with **single feature (TV)** using sklearn with train/test split

#### 🔑 Key Steps:

```
📥 Load Data: Advertising.csv
     ↓
🔍 Select Features: Keep only 'TV' column (drop radio, newspaper)
     ↓
📊 Set X, y: Independent (TV) and Dependent (sales) variables
     ↓
✂️ Split Data: 70% train, 30% test
     ↓
🤖 Train Model: LinearRegression().fit()
     ↓
🎯 Make Predictions: y_pred_slr on test set
     ↓
📈 Visualize: Actual vs Predicted scatter + line
     ↓
📊 Evaluate: R², MAE, MSE, RMSE metrics
```

#### 💡 Quick Memory:

- **Real-world approach** 🎲
- Train/Test split prevents overfitting
- Single independent variable
- Complete model evaluation

#### 🔢 Key Variables:

- `x` → TV (independent)
- `y` → sales (dependent)
- `slr` → LinearRegression model
- `y_pred_slr` → Predictions
- Metrics: MAE, MSE, RMSE

#### 📈 Evaluation Metrics:

| Metric   | Purpose                      |
| -------- | ---------------------------- |
| **R²**   | How well model fits (0-100%) |
| **MAE**  | Average prediction error     |
| **MSE**  | Squared average error        |
| **RMSE** | Square root of MSE           |

---

### 3️⃣ **MultipleLinearRegression.ipynb** - Multiple Linear Regression (sklearn)

#### 🎯 What it does:

Linear regression with **multiple features** (TV, radio, newspaper) using sklearn

#### 🔑 Key Steps:

```
📥 Load Data: Advertising.csv
     ↓
🔍 Select Features: TV, radio, newspaper (3 independent variables)
     ↓
📊 Set X, y: All features vs sales
     ↓
✂️ Split Data: 70% train, 30% test
     ↓
🤖 Train Model: LinearRegression().fit()
     ↓
📋 Show Results:
   - Intercept (b₀)
   - Coefficients for each feature (b₁, b₂, b₃)
     ↓
🎯 Make Predictions: y_pred_mlr on test set
     ↓
📈 Visualize: Actual vs Predicted scatter + perfect fit line
     ↓
📊 Evaluate: R², MAE, MSE, RMSE metrics
```

#### 💡 Quick Memory:

- **Advanced version** 🚀
- Multiple independent variables
- Shows how each feature contributes
- Better predictions than single variable
- Equation: `sales = b₀ + b₁×TV + b₂×radio + b₃×newspaper`

#### 🔢 Key Variables:

- `x` → TV, radio, newspaper (3 features)
- `y` → sales
- `mlr` → LinearRegression model
- `y_pred_mlr` → Predictions
- Model shows relationship of each feature to sales

---

## 🔄 Side-by-Side Comparison

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Feature         │ Regression.ipynb │ RegressionEX     │ MultipleLinRegr  │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Library         │ scipy.stats      │ sklearn          │ sklearn          │
│ Variables       │ 1                │ 1 (TV)           │ 3 (TV,radio,news)│
│ Train/Test      │ ❌ No            │ ✅ 70/30 split   │ ✅ 70/30 split   │
│ Visualization   │ ✅ Yes           │ ✅ Yes           │ ✅ Yes           │
│ Metrics         │ r value only     │ Full evaluation  │ Full evaluation  │
│ Use Case        │ Learning basics  │ Real project     │ Real project     │
│ Complexity      │ ⭐ Beginner      │ ⭐⭐ Intermediate│ ⭐⭐⭐ Advanced   │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## 🎓 The Complete Workflow

### Step 1: 📦 Import Libraries

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```

### Step 2: 📊 Load & Prepare Data

```python
dataset = pd.read_csv('Advertising.csv')
x = dataset[['TV']]  # or multiple columns
y = dataset['sales']
```

### Step 3: ✂️ Split Data (Train/Test)

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=100
)
```

### Step 4: 🤖 Train Model

```python
model = LinearRegression()
model.fit(x_train, y_train)
```

### Step 5: 🎯 Make Predictions

```python
y_pred = model.predict(x_test)
```

### Step 6: 📈 Visualize Results

```python
plt.scatter(x_test, y_test, label='Actual')
plt.plot(x_test, y_pred, 'red', label='Predicted')
plt.show()
```

### Step 7: 📊 Evaluate Model

```python
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(x, y) * 100
```

---

## 📌 Quick Formulas

### 🔢 Simple Linear Regression

$$y = b_0 + b_1 \times x$$

### 🔢 Multiple Linear Regression

$$y = b_0 + b_1 \times x_1 + b_2 \times x_2 + b_3 \times x_3 + ...$$

### 📊 Evaluation Metrics

- **R² Score**: $(1 - \frac{\sum(y_{actual} - y_{pred})^2}{\sum(y_{actual} - \bar{y})^2}) \times 100\%$
- **MAE**: $\frac{1}{n}\sum|y_{actual} - y_{pred}|$
- **MSE**: $\frac{1}{n}\sum(y_{actual} - y_{pred})^2$
- **RMSE**: $\sqrt{MSE}$

---

## 🎯 When to Use What?

| Scenario                            | Use This                         |
| ----------------------------------- | -------------------------------- |
| 📚 Learning regression basics       | `Regression.ipynb`               |
| 🏢 Real project with 1 feature      | `RegressionEX.ipynb`             |
| 🚀 Predicting with multiple factors | `MultipleLinearRegression.ipynb` |
| 🔬 Need highest accuracy            | `MultipleLinearRegression.ipynb` |
| ⚡ Quick prototype                  | `Regression.ipynb`               |

---

## 💾 Dataset: Advertising.csv

**Columns:**

- 📺 `TV` → TV advertising budget
- 📻 `radio` → Radio advertising budget
- 📰 `newspaper` → Newspaper advertising budget
- 💰 `sales` → Product sales (target variable)

---

## ✅ Key Takeaways

1. ✨ **Regression finds relationships** between input and output
2. 🎲 **Train/Test split** prevents overfitting
3. 📈 **More features** can improve accuracy
4. 📊 **Evaluation metrics** tell you how good your model is
5. 🔮 **Predictions** allow forecasting future values

---

## 🚀 Quick Start

1. Open any notebook
2. Run cells from top to bottom
3. Check visualization and metrics
4. Compare results between notebooks
5. Understand progression: simple → single feature → multiple features

---

**Created for EL 4152 - Machine Learning | Day 07 Practicals** 🎓
