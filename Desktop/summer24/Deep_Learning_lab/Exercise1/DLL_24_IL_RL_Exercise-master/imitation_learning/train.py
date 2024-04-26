import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

import sys
import torch

sys.path.append(".")
import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data", "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # Convert RGB images to grayscale
    X_train_gray = np.array([utils.rgb2gray(img) for img in X_train])
    X_valid_gray = np.array([utils.rgb2gray(img) for img in X_valid])
    
    # Discretize the action space
    y_train = np.array([utils.action_to_id(a) for a in y_train])
    y_valid = np.array([utils.action_to_id(a) for a in y_valid])

    # Normalize the image data to [0, 1]
    X_train_gray = X_train_gray / 255.0
    X_valid_gray = X_valid_gray / 255.0

    # Incorporate history if required
    if history_length > 1:
        # Function to stack images to create history
        def add_history(X, history_length):
            # Create a new array with a shape that includes history in the last dimension
            X_history = np.zeros((X.shape[0] - history_length + 1, X.shape[1], X.shape[2], history_length))
            for i in range(history_length):
                X_history[:, :, :, i] = X[i: X.shape[0] - history_length + 1 + i]
            return X_history
        
        # Apply the history function to training and validation data
        X_train_gray = add_history(X_train_gray, history_length)
        X_valid_gray = add_history(X_valid_gray, history_length)
        
        # Adjust labels to match the reduced number of samples due to history inclusion
        y_train = y_train[history_length - 1:]
        y_valid = y_valid[history_length - 1:]

    # Convert data to PyTorch tensors and add channel dimension
    X_train_tensor = torch.tensor(X_train_gray, dtype=torch.float32).unsqueeze(1)
    X_valid_tensor = torch.tensor(X_valid_gray, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, y_train, X_valid_tensor, y_valid

#from a help of internet
def sample_minibatch(X, y, batch_size):
    indices = np.random.choice(len(X), size=batch_size)
    return X[indices], y[indices]
#end help


def train_model(
    X_train,
    y_train,
    X_valid,
    n_minibatches,
    batch_size,
    lr,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    # agent = BCAgent(...)

    agent = BCAgent(learning_rate=lr)
    tensorboard_eval = Evaluation(tensorboard_dir, "Imitation Learning")

    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    #
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     if i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)

    for i in range(n_minibatches):
        X_batch, y_batch = sample_minibatch(X_train, y_train, batch_size)
        loss = agent.update(X_batch, y_batch)

        if i % 10 == 0:
            agent.net.eval()  # Set the network to evaluation mode for validation
            with torch.no_grad():
                val_predictions = agent.predict(X_valid)
                val_loss = agent.criterion(val_predictions, torch.tensor(y_valid, dtype=torch.long))
            if "val_loss" in tensorboard_eval.stats:
                tensorboard_eval.write_episode_data(i, {"loss": loss, "val_loss": val_loss.item()})

    # TODO: save your agent
    # model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    # print("Model saved in file: %s" % model_dir)
                
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)

if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./imitation_learning")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=1
    )


    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=1000, batch_size=64, lr=1e-4)

# %%
