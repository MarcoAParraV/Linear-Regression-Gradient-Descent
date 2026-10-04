# Linear Regression and Gradient Descent from Scratch

This notebook is a step-by-step study of simple linear regression and the main variants of Gradient Descent. The algorithms are implemented from scratch with NumPy so that every prediction, error calculation, and parameter update can be inspected directly.

The notebook focuses on understanding the learning process rather than hiding it behind a machine-learning library.

## Model

The model is a straight line:

```text
y_pred = m * X + b
```

where:

- `m` is the slope.
- `b` is the intercept or bias.
- `X` contains the input values.
- `y_pred` contains the predictions.

The model learns `m` and `b` by reducing the Mean Squared Error (MSE):

```text
MSE = mean((y - y_pred)²)
```

## Parameters and hyperparameters

`m` and `b` are **model parameters** because they are learned during training.

The following values are **hyperparameters** because they are selected before training and control how the model learns:

- `learning_rate`: size of each update.
- `epochs`: number of complete passes through the dataset.
- `batch_size`: number of observations used in each Mini-batch update.

The random seed is used to make the shuffled orders reproducible.

## Gradient Descent variants

### Stochastic Gradient Descent (SGD)

At the beginning of each epoch, the observation indices are shuffled. The algorithm visits every observation once in the resulting order and updates `m` and `b` immediately after each point.

```python
error = y_actual - y_pred

m = m + learning_rate * error * x_current
b = b + learning_rate * error
```

Because every observation produces an update, SGD learns quickly but its error curve can oscillate.

### Batch Gradient Descent

Batch Gradient Descent calculates the adjustment using the complete dataset and performs one update per epoch.

```python
predictions = m * X + b
errors = y - predictions

m_adjustment = np.mean(errors * X)
b_adjustment = np.mean(errors)

m = m + learning_rate * m_adjustment
b = b + learning_rate * b_adjustment
```

Its error curve is normally smoother, although each update becomes more expensive as the dataset grows.

### Mini-batch Gradient Descent

Mini-batch Gradient Descent shuffles the observations and divides them into small groups. The parameters are updated after processing each group.

It combines:

- More frequent updates than Batch Gradient Descent.
- More stable gradient estimates than pure SGD.
- Efficient computation for larger datasets and neural networks.

## Epochs and shuffling

One epoch is one complete pass through the training data. If the dataset contains 10 observations, one SGD epoch produces 10 parameter updates.

Shuffling changes the order of the observations at the beginning of every epoch:

```python
indices = list(range(len(X)))
random.shuffle(indices)
```

Every observation is still used exactly once during that epoch. A shuffled order may occasionally repeat in a later epoch; this does not invalidate the algorithm.

## Understanding the error curves

The notebook plots the MSE at the end of every epoch.

The SGD and Mini-batch curves are expected to fluctuate because their parameters are updated from individual observations or small groups. The order produced by shuffling affects the final position reached during each epoch.

Batch Gradient Descent normally produces a smoother curve because each update uses all observations.

A noisy curve does not necessarily mean that SGD is implemented incorrectly. Its overall tendency and its performance on unseen data are more informative than whether every epoch improves upon the preceding one.

## Final error versus best error

The final model contains the values of `m` and `b` obtained after the last epoch. With a constant learning rate, SGD may continue oscillating around the minimum, so an earlier epoch can have a lower training error.

For a real machine-learning workflow, the best model should normally be selected using a separate **validation set**, not the lowest error measured on the training data itself.

The usual data split is:

- **Training set:** updates the model parameters.
- **Validation set:** selects hyperparameters and the best epoch.
- **Test set:** evaluates the selected model once at the end.

## Final visualization

The notebook includes a comparison with two plots:

1. The original data and the final regression lines produced by SGD, Batch, and Mini-batch Gradient Descent.
2. The MSE curve across epochs for all three algorithms.

This comparison shows both the final fitted models and the different optimization behavior of each method.

## Requirements

- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

Install the dependencies with:

```bash
python -m pip install numpy matplotlib notebook
```

## Running the notebook

1. Clone or download the repository.
2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `Regresión_Lineal_Marco_Inglés.ipynb`.
4. Run the cells from top to bottom.

Running the notebook in order is important because the later cells use variables created by the dataset and training cells.

## Main learning outcomes

After completing the notebook, you should be able to explain:

- How a linear model makes predictions.
- How prediction error changes the slope and intercept.
- The difference between parameters and hyperparameters.
- The meaning of an epoch.
- Why training observations are shuffled.
- The difference between SGD, Batch, and Mini-batch Gradient Descent.
- Why stochastic error curves fluctuate.
- Why model selection should use validation data.

## Scope

This project is an educational implementation. For production applications, libraries such as scikit-learn, PyTorch, or TensorFlow provide optimized training algorithms, validation utilities, numerical safeguards, and support for larger datasets.
