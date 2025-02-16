import torch.nn as nn
import torch
class Generator(nn.Module):
    def __init__(self, seq_len, feature_dim):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        self.init_size = seq_len // 4  # Assuming the input time step is seq_len, first reduce it to seq_len/4.
        # Define convolution layers
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),  # (B, 1->32, T, N)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 2. Expand channel size
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 3. Further expansion
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 4. Output layer, recover to the input feature dimension
            nn.Conv2d(32, 1, kernel_size=(3, 3), stride=1, padding=1),  # (B, 1, T, N)
            # nn.Tanh()  # Normalize the output values to range [-1, 1]
        )

    def forward(self, z):
        B, T, N = z.shape  # Input shape (B, T, N)
        z = z.unsqueeze(1)  # Convert to (B, 1, T, N) since Conv2d requires a 4D input
        output = self.conv_blocks(z)  # Pass through the convolutional network
        output = output.squeeze(1)  # (B, T, N)
        return output


class Discriminator(nn.Module):
    def __init__(self, seq_len, feature_dim):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        self.model = nn.Sequential(
            # 1. Use convolution layers to process time and feature dimensions
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),  # (B, 32, T, N)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 2. Further feature extraction
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),  # (B, 64, T, N)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),  # Pooling layer, reducing spatial dimensions

            # 3. Further feature extraction
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),  # (B, 32, T/2, N/2)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 4, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),  # Pooling layer, reducing spatial dimensions # (B, 4, T/4, N/4)

            # 4. Flatten and progressively reduce dimensions with fully connected layers
            nn.Flatten(),  # (B, 4 * T/4 * N/4)
            nn.Linear(4 * (seq_len // 4) * (feature_dim // 4), 8192),  # Reduce dimensions progressively
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(8192, 4096),  # Intermediate layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(4096, 1024),  # Intermediate layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),  # Intermediate layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 256),  # Further reduction in dimensions
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),  # Final output
            nn.Sigmoid()  # Sigmoid function to get probability
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Convert (B, T, N) to (B, 1, T, N)
        validity = self.model(x)  # Compute the validity through the network
        return validity

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)  # Generate values from normal distribution N(mean,std)
        torch.nn.init.constant_(m.bias.data, 0.0)
