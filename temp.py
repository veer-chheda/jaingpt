import torch
import torch
import torch.nn as nn
import torch.optim as optim
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2

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

    input_size = 10
    hidden_size = 5
    output_size = 1
    # Define a loss function and an optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
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
    # Create an instance of the model
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(inputs)












