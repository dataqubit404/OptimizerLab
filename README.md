# OptimizerLab
### Gradient Descent vs RMSProp vs Adam Comparison

OptimizerLab is an interactive machine learning project that demonstrates and compares three popular optimization algorithms used in training machine learning models:

- Gradient Descent
- RMSProp
- Adam

The goal of this project is to help users understand how these optimizers behave during training and how quickly they converge to the minimum of a loss function.

---

## Project Overview

Optimization algorithms play a critical role in machine learning by minimizing the loss function and improving model performance.

This project provides a visual and interactive way to compare how different optimizers update parameters during training.

Users can run experiments and observe how the optimizers move toward the minimum point.

---

## Features

- Compare Gradient Descent, RMSProp, and Adam
- Interactive parameter control
- Visualization of optimizer behavior
- Loss vs Epoch comparison graphs
- Optimization path visualization
- Performance comparison table
- Simple and interactive UI

---

## How It Works

The system optimizes a function or trains a simple machine learning model using different optimizers.

### Steps

1. Initialize model parameters
2. Compute prediction
3. Calculate loss
4. Compute gradients
5. Update parameters using the selected optimizer
6. Track performance metrics
7. Display results using graphs and tables

---

## Optimizers Implemented

### Gradient Descent

Gradient Descent is the most basic optimization algorithm. It updates parameters in the direction of the negative gradient of the loss function.

Update rule:

θ = θ − α ∇J(θ)

Where:

θ = model parameters  
α = learning rate  
∇J(θ) = gradient of the loss function

---

### RMSProp

RMSProp adapts the learning rate for each parameter by dividing the gradient by a moving average of squared gradients.

This helps stabilize training and speeds up convergence.

---

### Adam

Adam (Adaptive Moment Estimation) combines the benefits of Momentum and RMSProp.

It keeps track of both:

- First moment (mean of gradients)
- Second moment (variance of gradients)

Adam is widely used in modern deep learning because of its fast convergence and stability.

---

## Visualizations

The project provides several visual outputs:

- Loss vs Epochs graph
- Optimizer convergence comparison
- Optimization path on function surface
- Final performance metrics table

These visualizations help users understand the behavior of each optimizer.

---

## Tech Stack

Programming Language:
- Python

Libraries:
- NumPy
- Matplotlib
- Pandas

Web Interface:
- Streamlit

Version Control:
- Git
- GitHub

---

-------------------------------------------------
OptimizerLab
Gradient Descent vs RMSProp vs Adam
-------------------------------------------------

Controls                     Results

Optimizer: [Compare All ▼]   Loss vs Epoch Graph

Learning Rate: [0.01]        Optimizer Path Graph

Epochs: [100]                Performance Table

Start Point: [8]

[ Run Simulation ]
[ Reset ]

---

## Project Structure

optimizerlab/
│
├── app.py
├── optimizers.py
├── model.py
├── visualization.py
├── requirements.txt
└── README.md

---

## Example Use Case

Users can experiment with different parameters such as:

- Learning rate
- Number of epochs
- Starting point

They can run simulations and observe how each optimizer converges toward the minimum.

---

## Results

Typical observations:

- Gradient Descent converges slowly
- RMSProp converges faster by adapting learning rates
- Adam converges fastest and provides stable results

---

## Future Improvements

Possible future enhancements include:

- Adding more optimizers
- Support for multiple datasets
- 3D loss surface visualization
- Neural network training experiments
- Hyperparameter tuning interface

---

## Author

Raj Darlami

---

## License

This project is developed for educational purposes.
