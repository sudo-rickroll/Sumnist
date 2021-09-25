from utils import data_loader, MNISTSumDataset, Process
from parsers import parse_config, parse_args
from models import MNIST_Sum_Model
from graphs import plot
import torch, torchvision

def main(args, config):
    train_set = MNISTSumDataset(root='./data', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), download=True)
    test_set = MNISTSumDataset(root='./data', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), download=True)
    train_loader = data_loader(train_set, int(config['DataLoader']['batch_size']))
    test_loader = data_loader(test_set, int(config['DataLoader']['batch_size']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNIST_Sum_Model(args.checkpoint_load).to(device)
    optimiser = getattr(torch.optim, config['Optimiser']['type'])(model.parameters(), float(config['Optimiser']['lr']))
    train_process = Process(int(config['Process']['epochs']), int(config['Process']['validate_per_epoch']), args.mode.lower(), model, device, config['Loss']['type'], train_loader, optimiser, test_loader, args.checkpoint_save)
    train_process.run()
    if 'train' in args.mode.lower():
      plot([(range(1, int(config['Process']['epochs']) + 1), train_process.accs_total['train']), (range(1, int(config['Process']['epochs']) + 1), train_process.accs_total['test'])], ['Training Set Accuracy', 'Testing Set Accuracy'], 'Epoch', 'Accuracy', 'Train and Test Accuracies')
      plot([(range(1, int(config['Process']['epochs']) + 1), train_process.losses_total['train']), (range(1, int(config['Process']['epochs']) + 1), train_process.losses_total['test'])], ['Training Set Loss', 'Testing Set Loss'], 'Epoch', 'Loss', 'Train and Test Losses')
      print("Images of the graphs stored in the 'images' folder")


if __name__ == '__main__':
   args = parse_args()
   config = parse_config(args.config_path) 
   main(args, config)
   