import pandas
from sklearn.preprocessing import MinMaxScaler

# this class creates scaled data from test & training data
# predictions work better if all data is in range 0 to 1
training_data_df = pandas.read_csv("sales_data_training.csv")

test_data_df = pandas.read_csv("sales_data_test.csv")

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_training_data = scaler.fit_transform(training_data_df)
# transform is called instead of fit_tranform so that same scaler value is used
scaled_test_data = scaler.transform(test_data_df)

# print the adjustment scaler used for future use
print("data was multiplied by {:.10f} and added {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

scaled_training_data_df = pandas.DataFrame(scaled_training_data, columns=training_data_df.columns.values)
scaled_test_data_df = pandas.DataFrame(scaled_training_data, columns=test_data_df.columns.values)

scaled_training_data_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_test_data_df.to_csv("sales_data_test_scaled.csv", index=False)

# data was multiplied by 0.0000036968 and added -0.115913
