from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('model/sites_models/lstm_2000.h5')

# Prepare your new input data (X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Shape it similarly to the training data

# Make predictions
predictions = model.predict(X_test)

# If you scaled your target data (y_train) during training, inverse transform the predictions
scaler = MinMaxScaler()
scaler.fit(y_train)  # Make sure to use the same scaling as during training
predictions_rescaled = scaler.inverse_transform(predictions)

# Print or use the predictions
print(predictions_rescaled)
