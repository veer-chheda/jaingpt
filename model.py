import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Define the input size, hidden layer size, and output size
input_size = 10
hidden_size = 5
output_size = 1

# Create an instance of the model
model = SimpleNN(input_size, hidden_size, output_size)

# Print the model architecture
print(model)

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create some random data for testing
inputs = torch.randn(10, input_size)
targets = torch.randn(10, output_size)

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Loss: {loss.item()}')