# Don't erase the template code, except "Your code here" comments.

import torch
import math                      # Pi
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

""" Task 1 """

def get_rho():
    # (1) Your code here; theta = ...
    theta = torch.linspace(-math.pi, math.pi, steps=1000, dtype=torch.float64)
    assert theta.shape == (1000,)

    # (2) Your code here; rho = ...
    rho = (1 + 0.9 * torch.cos(8 * theta)) \
          * (1 + 0.1 * torch.cos(24 * theta)) \
          * (0.9 + 0.05 * torch.cos(200 * theta)) \
          * (1 + torch.sin(theta))
    assert torch.is_same_size(rho, theta)

    # (3) Your code here; x = ...
    # (3) Your code here; y = ...
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    return x, y

""" Task 2 """


def game_of_life_update_torch(alive_map):
    """
    PyTorch version of `game_of_life_update_reference()`.

    alive_map:
        `torch.tensor`, ndim == 2, dtype == `torch.int64`
        The game map containing 0s (dead) an 1s (alive).
    """
    # Your code here

    # Convert to 4D tensor (to apply a convolution)
    inputs = alive_map.reshape(1, 1, alive_map.shape[0], alive_map.shape[1])

    # Create a kernel for the convolution
    kernel = torch.ones(1, 1, 3, 3)
    kernel[0, 0, 1, 1] = 0

    # Calculate #neighbors as convolution with zero_pad=1, stride=1  (h_out=h_in)
    num_of_neighbors = torch.conv2d(inputs.type(torch.LongTensor),
                                                  kernel.type(torch.LongTensor),
                                                  bias=None, stride=1, padding=1)

    # Alive (==1) when #neighbors==3 and when previously was alive and has 2 neighbors
    outputs = torch.zeros_like(inputs, dtype=torch.long)
    outputs = torch.where((num_of_neighbors == 3) | ((num_of_neighbors == 2) & (inputs == 1)), 1, outputs)

    # Output the result
    alive_map.data = outputs[0][0].clone().detach().data

""" Task 3 """

# This is a reference layout for encapsulating your neural network. You can add arguments and
# methods if you need to. For example, you may want to add a method `do_gradient_step()` that
# executes one step of an optimization algorithm (SGD / Adadelta / Adam / ...); or you can
# add an extra argument for which you'll find a good value during experiments and set it as
# default to preserve interface (e.g. `def __init__(self, num_hidden_neurons=100):`).
class NeuralNet:
    def __init__(self, n_in=28*28, n_hidden=200, n_classes=10):
        # Your code here
        stdv = 1. / math.sqrt(n_in)

        self.W1 = torch.normal(0, stdv, size=(n_in, n_hidden), requires_grad=True)
        self.b1 = torch.normal(0, stdv, size=(1, n_hidden), requires_grad=True)
        self.W2 = torch.normal(0, stdv, size=(n_hidden, n_classes), requires_grad=True)
        self.b2 = torch.normal(0, stdv, size=(1, n_classes), requires_grad=True)
        self.learning_rate = 0.01

    def predict(self, images):
        """
        images:
            `torch.tensor`, shape == `batch_size x height x width`, dtype == `torch.float32`
            A minibatch of images -- the input to the neural net.
        
        return:
        prediction:
            `torch.tensor`, shape == `batch_size x 10`, dtype == `torch.float32`
            The scores of each input image to belong to each of the dataset classes.
            Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
            belong to `j`-th class.
            These scores can be 0..1 probabilities, but for better numerical stability
            they can also be raw class scores after the last (usually linear) layer,
            i.e. BEFORE softmax.
        """
        # Your code here

        # Reshape 2D data to 1D
        X = images.reshape(images.shape[0], -1)

        self.linear1 = X @ self.W1 + self.b1
        self.nonlinearity = torch.maximum(torch.tensor((0,)), self.linear1)
        self.linear2 = self.nonlinearity @ self.W2 + self.b2
        self.output = torch.exp(self.linear2 - torch.max(self.linear2)) / torch.exp(self.linear2 - torch.max(self.linear2)).sum(dim=1, keepdim=True)
        return self.output

    # Your code here
    def do_gradient_step(self):
        with torch.no_grad():
            self.W1.copy_(self.W1 - self.learning_rate * self.W1.grad.detach())
            self.W2.copy_(self.W2 - self.learning_rate * self.W2.grad.detach())
            self.b1.copy_(self.b1 - self.learning_rate * self.b1.grad.detach())
            self.b2.copy_(self.b2 - self.learning_rate * self.b2.grad.detach())

    def zeroGradParameters(self):
        self.W1.grad.zero_()
        self.W2.grad.zero_()
        self.b2.grad.zero_()
        self.b1.grad.zero_()


def accuracy(model, images, labels):
    """
    Use `NeuralNet.predict` here.
    
    model:
        `NeuralNet`
    images:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    labels:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `images`.
    
    return:
    value:
        `float`
        The fraction of samples from `images` correctly classified by `model`.
        `0 <= value <= 1`.
    """
    # Your code here

    # One-hot y_train
    y_hot = torch.eye(10)[labels]

    batch_size = 1500
    y_vals = []
    y_predictions = []
    for x_batch, y_batch in get_batches((images, y_hot), batch_size):
        predictions = model.predict(x_batch)
        y_predictions = y_predictions + list(predictions.argmax(dim=1))
        y_vals = y_vals + list(y_batch.argmax(dim=1))

    acc = accuracy_score(y_vals, y_predictions)
    return acc

def train_on_notmnist(model, X_train, y_train, X_val, y_val):
    """
    Update `model`'s weights so that its accuracy on `X_val` is >=82%.
    `X_val`, `y_val` are provided for convenience and aren't required to be used.
    
    model:
        `NeuralNet`
    X_train:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    y_train:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `X_train`.
    X_val, y_val:
        Same as above, possibly with a different length.
    """
    # Your code here

    # One-hot y_train
    y_train_hot = torch.eye(10)[y_train]

    # Hyperparameters
    n_epochs = 15
    batch_size = 1000

    loss_history = []
    loss_history_val = []
    for i in range(n_epochs):
        loss_epoch = []
        for x_batch, y_batch in get_batches((X_train, y_train_hot), batch_size):
            # Forward step
            predictions = model.predict(x_batch)

            # Calculate loss
            predictions_clamp = torch.clamp(predictions, 1e-15, 1 - 1e-15) # trick to avoid numerical errors
            loss = - torch.sum(y_batch * torch.log(predictions_clamp)) / y_batch.shape[0]

            # Backward step
            loss.backward()

            # Stochastic Gradient Descent step
            model.do_gradient_step()

            # Update history of losses
            loss_epoch.append(loss.item())

            # Zero gradients
            model.zeroGradParameters()

        loss_history.append(sum(loss_epoch) / len(loss_epoch))
        print(f'Epoch {i}, train loss = {loss_history[-1]}')

    plt.plot(range(1, n_epochs + 1), loss_history, label='train')
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    print(f'Train accuracy: {accuracy(model, X_train, y_train) * 100} %')
    print(f'Validation accuracy: {accuracy(model, X_val, y_val) * 100} %')




def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]

    # Shuffle at the start of epoch
    indices = torch.randperm(n_samples)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_idx = indices[start:end]

        yield X[batch_idx], Y[batch_idx]

