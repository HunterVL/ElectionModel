import numpy as np
from itertools import product
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import regularizers, optimizers, losses
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

X = np.load('Outputs/X.npy')
y = np.load('Outputs/y.npy')

# create an RNN with specified hyperparameters
def create_RNN(activation, hidden_units, learning_rate, reg_rate):
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    regularizer = regularizers.L2(l2=reg_rate)

    model = Sequential()
    model.add(LSTM(
        hidden_units, 
        input_shape=X[0].shape,
        activation=activation,
        kernel_regularizer=regularizer
    ))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

# Partition the dataset into training and validation sets
val_label = np.random.uniform(0, 1, X.shape[0]) < 0.3
X_val = X[val_label]
y_val = y[val_label]
X_train = X[~val_label]
y_train = y[~val_label]

# Check the loss for a simple average of recent polls.
# The model should have a lower loss.
y_pred_naive = X_val[:,-10:,2].mean(axis=1)
bce = losses.BinaryCrossentropy(from_logits=False)
loss_naive = bce(y_val, y_pred_naive)
print(f"Simple loss: {loss_naive}\n")

# Define the hyperparameters to be checked in cross-validation.
hyperparams_activation = ["tanh", "leaky_relu", "relu"]
hyperparams_units = [5, 10, 20]
hyperparams_alpha = [1e-5, 3e-5, 1e-4]
hyperparams_lambda = [0, 1e-6, 1e-5]
cartProd = product(hyperparams_activation, hyperparams_units, hyperparams_alpha, hyperparams_lambda)
batch_size = 200
best_val_loss = np.inf
best_hyperparams = None

# Find the best hyperparameters
for activation, units, alpha, lambda_val in cartProd:
    model = create_RNN(activation, units, alpha, lambda_val)
    seqModel = model.fit(
        X_train, y_train, 
        batch_size=batch_size,
        epochs=500, 
        verbose=0, # type: ignore
        validation_data = (X_val, y_val)
    )

    val_loss = seqModel.history['val_loss'][0]
    print(f"Activation function: {activation}\n"+
          f"Hidden layer units: {units}\n"+
          f"Learning rate: {alpha}\n"+
          f"Regularization rate: {lambda_val}\n"+
          f"Validation loss: {val_loss}\n")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_hyperparams = activation, units, alpha, lambda_val

# Train the model on the full dataset using the best hyperparameters
model = create_RNN(*best_hyperparams)
model.fit(
    X, y, 
    batch_size=batch_size,
    epochs=5000,
    verbose=0, # type: ignore
)

model.save("Outputs/my_model.keras")