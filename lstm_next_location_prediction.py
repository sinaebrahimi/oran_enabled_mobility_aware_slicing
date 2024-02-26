import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Assume we have some preprocessed data
# self.loc_user_init = np.zeros([T, USER_NO, 2])
T = 20
USER_NO = 2
self.loc_user_init = np.random.rand(T, USER_NO, 2)  # Just for the sake of example

# We will use a sequence length of 3 for this example
sequence_length = 3

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, USER_NO * 2))) # 50 is the number of LSTM units (or neurons) in that layer
model.add(Dense(USER_NO * 2))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Loop over each timestep
for t in range(sequence_length, T):
    # Prepare the sequences and the corresponding labels
    data = self.loc_user_init[t-sequence_length:t]
    labels = self.loc_user_init[t]

    # Convert data and labels to numpy arrays
    data = np.array([data])
    labels = np.array([labels])

    # Train the model
    model.fit(data, labels, epochs=200, verbose=0)

    # Now you can use model.predict to predict the locations for the next timestep
    next_location = model.predict(np.array([self.loc_user_init[t-sequence_length+1:t+1]]))
