# 🚀 Deep Learning Optimizer Comparison: SGD vs. RMSProp vs. Adam

This project serves as an empirical study on the performance, convergence speed, and stability of different optimization algorithms in Deep Learning. Using a **Linear Regression** architecture built with **PyTorch**, we predict housing prices using the real-world **California Housing Dataset**.

---

## 📌 1. Objective
The primary goal is to demonstrate how different optimizers navigate the loss landscape. We compare:
* **Stochastic Gradient Descent (SGD):** The baseline optimization algorithm.
* **RMSProp:** An adaptive learning rate method that tackles diminishing gradients.
* **Adam:** A combination of momentum and adaptive scaling, often considered the industry standard.

---

## 📊 2. Dataset Used
**California Housing Dataset** (Scikit-Learn)
* **Features:** 8 numerical features including `MedInc` (Median Income), `HouseAge`, `AveRooms`, etc.
* **Target:** Median house value for California districts (expressed in hundreds of thousands of dollars).
* **Size:** 20,640 samples.

---

## 🧠 3. Mathematical Concept

The model minimizes the **Mean Squared Error (MSE)** loss function:
$$J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Optimizers Compared:
1. **SGD:** Updates parameters by moving in the opposite direction of the gradient.
   $$\theta_{t+1} = \theta_t - \eta \cdot \nabla J(\theta_t)$$
2. **RMSProp:** Scales the learning rate by a moving average of the squared gradients.
   $$v_t = \beta v_{t-1} + (1-\beta)g_t^2 \implies \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}}g_t$$
3. **Adam:** Combines the advantages of both Adagrad and RMSProp using first and second moment estimates.
   $$\theta_{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

## ⚙️ 4. Methodology & Project Steps

### Step 1: Environment Setup
* Install Python 3.9+.
* Install dependencies: `torch`, `streamlit`, `scikit-learn`, `pandas`, `matplotlib`.

### Step 2: Data Preprocessing
* Load data and perform **Standardization** (scaling features to zero mean and unit variance). This is critical for gradient-based optimizers to converge effectively.

### Step 3: Architecture
* Define a single-layer Linear Regression model using `torch.nn.Module`.

### Step 4: Training Loop
* Train the model three separate times using identical initial weights but different optimizers.
* Track the loss at every epoch for comparison.

### Step 5: Visualization & Deployment
* Use **Streamlit** to create an interactive dashboard for hyperparameter tuning (learning rate, epochs).
* Plot loss curves to visualize convergence.

---

## 📈 5. Results & Results Analysis
Based on the experiments conducted:

Adam typically shows the fastest convergence, reaching the lowest MSE in fewer than 100 epochs.

RMSProp is stable and avoids the oscillations often seen in vanilla SGD.

SGD requires a much smaller learning rate to avoid "exploding gradients" and takes longer to reach the global minimum.

---

## 🎯 6. Conclusion
This project highlights that while traditional Gradient Descent is the foundation of deep learning, adaptive optimizers like Adam and RMSProp are significantly more efficient for high-dimensional, real-world datasets. For house price prediction, Adam provides the most robust and rapid path to a localized minimum.

---


## 🛠️ 7. Built With
PyTorch - Deep Learning Framework
Streamlit - Web App Framework
Scikit-Learn - Machine Learning Tools

---

## 8. Tech Stack

Programming Language:
- Python

Libraries & Frameworks
- NumPy
- Matplotlib
- Pandas
- Pytorch

Web Interface:
- Streamlit

Version Control:
- Git
- GitHub

---

