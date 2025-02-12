# !pip install sklearn

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target# Convert X and y to Pandas DataFrames
X_df = pd.DataFrame(X, columns=data.feature_names)
# Display the loaded data
print("California Housing Data:")
print(X_df.head())

y_df = pd.DataFrame(y, columns=["target"])
print("\nCalifornia Housing Target:")
print(y_df.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)
print(X_train)

# Define a simple regression model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(10,1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# Create the model
input_size = X_train.shape[1]
print(input_size)
model = RegressionModel(input_size)
print(model)

# Define loss and optimizer
Lossf = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = Lossf(outputs, y_train.view(-1, 1))
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'california_housing_model.pth')

# Load the model for future use
loaded_model = RegressionModel(input_size)
loaded_model.load_state_dict(torch.load('california_housing_model.pth'))

# Evaluate the loaded model on the test set
with torch.no_grad():
    y_pred = loaded_model(X_test)
    mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
    print(f'Mean Squared Error on Test Data (Loaded Model): {mse:.4f}')
