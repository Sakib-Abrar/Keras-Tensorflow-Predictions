import pandas
from keras import Sequential
from keras.layers import Dense

# Load training data
training_data_df = pandas.read_csv("../data/sales_data_training_scaled.csv")

training_input = training_data_df.drop('total_earnings', axis=1).values
training_output = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
model.fit(
    training_input,
    training_output,
    epochs=50,
    shuffle=True,
    verbose=2
)

# Load test data
test_data_df = pandas.read_csv("../data/sales_data_test_scaled.csv")

test_input = test_data_df.drop('total_earnings', axis=1).values
test_output = test_data_df[['total_earnings']].values

# evaluate trained model
test_error_rate = model.evaluate(test_input, test_output, verbose=0)
print("The mean squared error of the trained model for test data is {}".format(test_error_rate))

# save model
model.save("trained_model.h5")
