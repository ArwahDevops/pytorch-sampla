import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Read the CSV file and extract the relevant columns
df = pd.read_csv('filtered_data.csv')
data = df[['Hari','Jam','Result']]

# Encode the categorical feature 'Hari'
le = LabelEncoder()
data['Hari'] = le.fit_transform(data['Hari'])
ohe = OneHotEncoder(handle_unknown='ignore')
data = ohe.fit_transform(data[['Hari']]).toarray()
data = pd.DataFrame(data)

# split data into input and output
x_data = data.values
y_data = df['Result'].values

# Define the neural network
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size).float()

    def forward(self, x):
        out = self.linear(x)
        return out

# Define the model, loss function, and optimizer
input_size = len(x_data[0])
output_size = 1
model = LinearRegression(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_data).float()
    labels = torch.from_numpy(y_data).float()

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)

    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

# Make predictions
with torch.no_grad():
    x_test = torch.from_numpy(x_data).float()
    y_test = model(x_test)
    y_test = y_test.numpy()
    
# Print the predictions
print(y_test)
