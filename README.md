# 🏠 ML Model Comparison for House Price Prediction

A powerful, interactive Streamlit web application that compares multiple machine learning models for predicting California house prices. It provides detailed model performance metrics, visualizations, and complexity analysis to help users understand trade-offs between different algorithms.

## 🚀 Live Demo

> **Note**: You can run the app locally by following the instructions below.

---

## 📦 Features

- 📊 **Dataset Overview** — Explore the California Housing dataset.
- 🔧 **Model Training** — Train Linear Regression, Random Forest, Gradient Boosting, and SVR.
- 📈 **Results & Comparison** — Visual comparison of model metrics, predictions, and performance.
- 🧮 **Complexity Analysis** — Theoretical analysis of time/space complexity and algorithm interpretability.

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Plotly](https://plotly.com/)
- [Matplotlib](https://matplotlib.org/)
- [psutil](https://pypi.org/project/psutil/)

---

## 📥 Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/An1rud/ML-Model-Comparison-for-House-Price-Prediction.git
cd ML-Model-Comparison-for-House-Price-Prediction
pip install -r requirements.txt
````

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the app in your browser at `http://localhost:8501`.

---

## 📊 Models Compared

| Model             | Type                | Highlights                          |
| ----------------- | ------------------- | ----------------------------------- |
| Linear Regression | Linear Model        | Fast, interpretable, baseline       |
| Random Forest     | Ensemble (Bagging)  | Handles non-linear patterns, robust |
| Gradient Boosting | Ensemble (Boosting) | High accuracy, risk of overfitting  |
| SVR (RBF Kernel)  | Kernel-based Method | Effective in high dimensions        |

---

## 📈 Metrics Evaluated

* **R² Score**
* **Mean Squared Error (MSE)**
* **Mean Absolute Error (MAE)**
* **Training Time**
* **Prediction Time**
* **Memory Usage**

---

## 📚 Dataset

**California Housing Dataset**
Built-in with scikit-learn. It includes:

* 20,640 samples
* 8 features (e.g., population, income, house age)
* Target: Median house price

---

## 📎 Project Structure

```
.
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📜 License

MIT License © [An1rud](https://github.com/An1rud)
