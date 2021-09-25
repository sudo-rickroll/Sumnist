import torch
import torch.nn as nn

class MNIST_Sum_Model(nn.Module):
  def __init__(self, checkpoint_path):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(32)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )

    self.transition = nn.Sequential(
        nn.MaxPool2d(2,2),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
    )
    
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )

    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc1_1 = nn.Linear(1*1*64, 20)
    self.fc1_2 = nn.Linear(20, 10)
    self.fc2_1 = nn.Linear(74, 30)
    self.fc2_2 = nn.Linear(30, 19)
    
    if checkpoint_path:
        print(f"Loading model checkpoint from '{checkpoint_path}'")
        try:
          parameters = torch.load(checkpoint_path)['model']    
          model_dict = self.state_dict()
          load_dict = {key: item for key, item in parameters.items() if key in model_dict}
          model_dict.update(load_dict)
          self.load_state_dict(model_dict)
          print("Model loaded successfully")
        except:
          print('Could not load model')

  def forward(self, image, rand):
    image = self.conv4(self.conv3(self.transition(self.conv2(self.conv1(image)))))
    image = self.avg_pool(image)
    image_comb = image.reshape(-1, 1*1*64)
    image = self.fc1_2(self.fc1_1(image_comb))

    rand = nn.functional.one_hot(torch.tensor(rand), num_classes=10)
    rand = torch.cat((image_comb, rand), dim=1)
    rand = self.fc2_2(self.fc2_1(rand))

    cls1 = torch.nn.functional.log_softmax(image, dim=-1)
    cls2 = torch.nn.functional.log_softmax(rand, dim=-1)

    return cls1, cls2