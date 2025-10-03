import torch
import torch.nn as nn
import shutil


# Residual Block Class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut  # Add the shortcut connection
        x = self.relu(x)
        return x

# Main ActorCriticResNet Model
class ActorCriticResNet(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(ActorCriticResNet, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Actor (Policy) network
        self.actor_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Ensures output is in the range [-1, 1]
        )

        # Critic (Value) network
        self.critic_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    # def policy(self, x):
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.avgpool(x)
    #     x = self.flatten(x)

    #     policy_logits = self.actor_fc(x) 
    #     return policy_logits, value

    # def value(self, x):
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.avgpool(x)
    #     x = self.flatten(x)
 
    #     value = self.critic_fc(x)
    #     return   value



    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        policy_logits = self.actor_fc(x)
        value = self.critic_fc(x)
        return policy_logits, value
    

    def save_ckp(self, state, is_best, checkpoint_dir, name):
        f_path = f"{checkpoint_dir}/{name}"
        torch.save(state, f_path)
        if is_best:
            best_fpath = f"{checkpoint_dir}/best_model.pt"
            shutil.copyfile(f_path, best_fpath)


    def load_ckp(self, checkpoint_fpath, model):
        checkpoint = torch.load(checkpoint_fpath, weights_only=True)
        model.load_state_dict(checkpoint['state_dict']) 
        return model,  checkpoint['epoch']
  

# Testing the Model
if __name__ == "__main__":
    # Define input parameters
    batch_size = 4  # Number of images in a batch
    input_channels = 3  # RGB image
    image_height = 224  # Image height
    image_width = 224  # Image width
    action_dim = 2  # Number of actions (e.g., left and right wheel velocities)

    # Initialize the model
    model = ActorCriticResNet(input_channels=input_channels, action_dim=action_dim)

    # Create a dummy input tensor (random image data)
    dummy_input = torch.randn(batch_size, input_channels, image_height, image_width)

    # Pass the input through the model
    policy_logits, value = model(dummy_input)

    # Print the outputs
    print("Policy Logits (Action Outputs):", policy_logits)
    print("Value (Critic's Evaluation):", value)
