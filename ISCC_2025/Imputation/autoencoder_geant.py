import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Data loading and preprocessing
df = pd.read_csv('../datasets/net_traffic/GEANT/geant.csv', header=None)  # Read dataset (please replace with the actual path)
df = df.iloc[:5000]

data = df.values
data = data / 1e7

# Convert to PyTorch Tensor
data_tensor = torch.tensor(data, dtype=torch.float32)

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data_tensor, test_size=0.2, random_state=42)


# 2. Build the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Use Sigmoid to ensure the output values are between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 3. Model initialization and training setup
input_dim = 529  # Number of features
hidden_dim = 128  # Hidden layer dimension
model = Autoencoder(input_dim, hidden_dim)

# Set the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Use Mean Squared Error loss

# 4. Train the model and save the model with the minimum validation loss
num_epochs = 1000
batch_size = 64
best_loss = float('inf')  # Set the initial validation loss to infinity
best_model = None  # Save the model with the minimum validation loss

early_stop = 0
patience = 8
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(train_data), batch_size):
        batch_x = train_data[i:i + batch_size]

        # Forward propagation
        reconstructed = model(batch_x)

        # Calculate the loss
        loss = criterion(reconstructed, batch_x) * 1000

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate at the end of each epoch
    model.eval()
    with torch.no_grad():
        val_reconstructed = model(val_data)
        val_loss = criterion(val_reconstructed, val_data) * 1000

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    if early_stop >= patience:
        print("Early stopping")
        break
    # Update the best model
    if val_loss < best_loss:
        print(f'best_loss:{best_loss}->{val_loss}')
        early_stop = 0
        best_loss = val_loss
        best_model = model.state_dict()  # Save the state dictionary of the best model
    else:
        early_stop += 1;

# 5. Reconstruct the entire dataset using the best model
model.load_state_dict(best_model)  # Load the best model
model.eval()

with torch.no_grad():
    reconstructed_data = model(data_tensor)

# 6. Calculate the error and remove the top 5% of the rows with the highest error
# Calculate the Mean Squared Error (MSE) for each row
mse = torch.mean((data_tensor - reconstructed_data) ** 2, dim=1)

# Find the rows with the highest 5% error
error_threshold = torch.quantile(mse, 0.9)  # Find the threshold for the top 5% errors
indices_to_remove = mse > error_threshold  # Get the indices of the rows with error greater than the threshold

# Output the indices of the deleted rows
deleted_indices = indices_to_remove.nonzero(as_tuple=True)[0].tolist()
print(len(deleted_indices))
print(f"Deleted row indices: {deleted_indices}")
