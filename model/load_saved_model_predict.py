import pandas
from keras.models import load_model

prediction_input = pandas.read_csv("../data/proposed_new_product.csv")

# Load saved model and predict
model = load_model("trained_model.h5")

prediction = model.predict(prediction_input)

# Adjust for scaling
prediction = prediction + 0.115913
prediction = prediction / 0.0000036968

print("The total predicted earning of the proposed prodcut is ${}".format(prediction[0][0]))
