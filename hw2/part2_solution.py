# Don't erase the template code, except "Your code here" comments.

import torch
# Your code here...

# # TensorBoard
# %load_ext tensorboard
#
# import os
# logs_base_dir = "./logs"
# os.makedirs(logs_base_dir, exist_ok=True)
# %tensorboard --logdir {logs_base_dir}

import torchvision
import numpy as np
from matplotlib import pylab

from datetime import datetime

# # TensorBoard
# # LOADING LOGGER (SummaryWriter)
# from torch.utils.tensorboard import SummaryWriter

import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from IPython import display
import numpy as np
from sklearn.metrics import accuracy_score


def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.

    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here

    data_transform = {'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    shuffle_option = {'train': True,
                      'val': False
                      }

    batch_size = 64

    tiny_imagenet = datasets.ImageFolder(root=(path + kind),
                                         transform=data_transform[kind])
    dataset_loader = torch.utils.data.DataLoader(tiny_imagenet,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle_option[kind],
                                                 num_workers=2,
                                                 pin_memory=True,
                                                 )
    return dataset_loader


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        num_classes = 200
        n_channels = 3

        self.cnn_layers = nn.Sequential(
            # Defining a Conv2d layer
            nn.Conv2d(n_channels, 20, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # h_out = 65
            # Defining another Conv2d layer
            nn.Conv2d(20, 50, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # h_out = 32
            # Defining another Conv2d layer
            nn.Conv2d(50, 100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # h_out = 31
            # Defining another Conv2d layer
            nn.Conv2d(100, 200, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # h_out = 15
            # Defining another Conv2d layer
            nn.Conv2d(200, 300, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # h_out = 14
            # Defining another Conv2d layer
            nn.Conv2d(300, 400, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # h_out = 7
            # Defining another Conv2d layer
            nn.Conv2d(400, 500, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(500),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # h_out = 3
        )

        self.linear_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(500 * 3 * 3, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, num_classes),
        )

    # The forward pass:
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    # Your code here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyNet().to(device)

    return model


def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # Your code here
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
    return optimizer


def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.A
    """
    # Your code here
    loss_function = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)

    n_epochs = 100

    losses_across_epochs = []

    val_epochs = []  # numbers of epochs when the validation metric is calculated
    val_accuracys = []
    val_losses = []
    best_val_accuracy = 0.0

    # # TensorBoard
    # # ---
    # # INITIALIZING SummaryWriter OBJECT
    # # ...
    # exp_name = datetime.now().isoformat(timespec='seconds')
    # writer = SummaryWriter(log_dir=f'logs/{exp_name}')
    # # ---

    for epoch in tqdm(range(n_epochs)):

        loss_curr_epoch = []

        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            # Get labels and images in current batch
            x_batch, y_batch = batch

            # Zero gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = model(x_batch.to(device))
            loss = loss_function(y_pred, y_batch.to(device))
            loss.backward()
            optimizer.step()

            # Append currrent loss after one batch
            loss_curr_epoch.append(loss.item())

            # Display loss every 300 batches (starting from 300th)
            if i % 300 == 299:
                print(f'[{epoch + 1}, {i + 1}] loss: {(np.mean(loss_curr_epoch)):.3}')

        losses_across_epochs.append(np.mean(loss_curr_epoch))

        # Change step of optimizer (if needed)
        scheduler.step()

        # Check accuracy on validation every 2nd epoch (starting with the first) and at the last epoch
        if epoch % 2 == 0 or epoch == (n_epochs - 1):
            val_acc, val_loss = validate(val_dataloader, model)
            val_epochs.append(epoch + 1)
            val_accuracys.append(val_acc)
            val_losses.append(val_loss)

            # If accuracy on validation set is better, then save this checkpoint
            if val_acc >= best_val_accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses_across_epochs[-1],
                },
                    './checkpoint.pth')
                best_val_accuracy = val_acc

            # # TensorBoard
            # # SENDING LOSS TO TENSORBOARD
            # writer.add_scalar('validation loss', val_loss, global_step=epoch + 1)
            # writer.add_scalar('validation accuracy', val_acc, global_step=epoch + 1)

        # # TensorBoard
        # # SENDING LOSS TO TENSORBOARD
        # writer.add_scalar('train loss', losses_across_epochs[-1], global_step=(epoch + 1))

        display.clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
        ax1.plot(range(1, len(losses_across_epochs) + 1), losses_across_epochs, label='train')
        ax1.plot(val_epochs, val_losses, label='val')
        ax1.set_xlabel('# epochs')
        ax1.set_ylabel('loss')
        ax1.set_title(f'Current train loss {losses_across_epochs[-1]:.4f}, last val loss {val_losses[-1]:.4f}')
        ax1.legend()

        ax2.plot(val_epochs, val_accuracys, label='val')
        ax2.set_xlabel('# epochs')
        ax2.set_ylabel('accuracy, in decimals')
        ax2.set_title(f'Best accuracy {(best_val_accuracy * 100):.3f}, Last accuracy {(val_accuracys[-1] * 100):.3f}')
        ax2.legend()

        plt.show()


@torch.no_grad()
def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # обнуляем веса
    model.eval()

    # forward (batch == x_batch == images)
    y_pred = model(batch.to(device))

    return y_pred


def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here
    loss_function = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    y_vals = []
    y_predictions = []

    losses = []

    for batch in tqdm(dataloader):
        # Get labels and images in current batch
        x_batch, y_batch = batch
        y_pred_batch = predict(model, x_batch)

        # Calculate loss on this batch and save it
        loss_batch = loss_function(y_pred_batch.to(device), y_batch.to(device))
        losses.append(loss_batch.item())

        # Calculate labels on this batch and save them (as list)
        _, label_batch = y_pred_batch.max(1)
        y_predictions.extend(label_batch.tolist())
        y_vals.extend(y_batch.tolist())

    # Calculate loss as mean over batches' losses and accuracy score
    loss = np.mean(losses)
    acc_score = accuracy_score(y_predictions, y_vals)

    return acc_score, loss


def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])


def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here;
    md5_checksum = "7d4bc46dfb2abd5255d26a52e01905e4"
    # Your code here;
    google_drive_link = "https://drive.google.com/file/d/1-RH-ohBLfukDD72mcHCgs3uwIWHg20XG/view?usp=sharing"

    return md5_checksum, google_drive_link
