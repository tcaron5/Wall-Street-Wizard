import os
import sys
import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episode_loss = 0

action_size = 3  # buy, sell, hold
inv = []
memory = deque(maxlen=1000)

def model_init(state_size, action_size):
    """
    Initializes a neural network model for reinforcement learning.
    
    Parameters:
    - state_size (int): The size of the input layer, representing the state space.
    - action_size (int): The size of the output layer, representing the number of possible actions.
    
    Returns:
    - model (Sequential): The compiled Keras model ready for training.
    """
    model = Sequential()
    model.add(Dense(units=64, input_dim=state_size, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=8, activation="relu"))
    model.add(Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
    return model

def act(model, state, epsilon, action_size):
    """
    Selects an action using an epsilon-greedy policy.
    
    Parameters:
    - model (Sequential): The trained model to predict Q-values.
    - state (ndarray): The current state of the environment.
    - epsilon (float): The exploration-exploitation trade-off parameter.
    - action_size (int): The number of possible actions.
    
    Returns:
    - action (int): The chosen action index.
    """
    if random.random() <= epsilon:
        return random.randrange(action_size)
    opts = model.predict(state)
    return np.argmax(opts[0])

def exp_replay(mem, model, batch_size, gamma, epsilon, epsilon_min, epsilon_decay):
    """
    Performs experience replay to train the model on past experiences.
    
    Parameters:
    - mem (deque): The memory buffer storing past experiences.
    - model (Sequential): The neural network model to be trained.
    - batch_size (int): The number of samples to draw from memory for training.
    - gamma (float): The discount factor for future rewards.
    - epsilon (float): The current exploration rate.
    - epsilon_min (float): The minimum value for epsilon.
    - epsilon_decay (float): The factor by which epsilon is multiplied after each training step.
    
    Returns:
    - mean_loss (float): The average loss over the training batch.
    """
    if len(mem) < batch_size:
        # Not enough data to process
        return 0  
    
    small_batch = random.sample(memory, batch_size)
    losses = []

    for state, action, reward, next_state, done in small_batch:
        goal = reward
        if not done:
            goal = reward + gamma * np.amax(model.predict(next_state)[0])
        goal_f = model.predict(state)
        goal_f[0][action] = goal
        loss = model.fit(state, goal_f, epochs=1, verbose=0).history['loss'][0]
        losses.append(loss)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    return np.mean(losses) if losses else 0

def find_latest_checkpoint(model_dir):
    """
    Finds the latest model checkpoint in the given directory.
    
    Parameters:
    - model_dir (str): The directory where model checkpoints are saved.
    
    Returns:
    - latest_checkpoint (int or None): The latest checkpoint number, or None if no checkpoints are found.
    """
    checkpoint_files = [int(f.split('.')[0]) for f in os.listdir(model_dir) if f.endswith('.keras')]
    return max(checkpoint_files) if checkpoint_files else None
def stockVector(key):
    """
    Reads stock data from a CSV file and extracts the closing prices.
    
    Parameters:
    - key (str): The file name (without extension) of the CSV file containing stock data.
    
    Returns:
    - v (list of float): A list of closing prices extracted from the CSV file.
    """
    lines = open(key+".csv","r").read().splitlines()
    v = []
    for line in lines[1:]:
        v.append(float(line.split(",")[4]))
    return v

def priceFormat(p):
    """
    Formats a number as a price string with a dollar sign and two decimal places.
    
    Parameters:
    - n (float): The number to be formatted as a price.
    
    Returns:
    - formatted_price (str): The formatted price string.
    """
    if p < 0:
        sign = "-$"
    else:
        sign = "$"
    
    formatted_price = "{0:.2f}".format(abs(p))
    return sign + formatted_price

def getState(price_data, current_time, window_size):
    """
    Constructs the state representation for the trading agent based on historical stock prices.
    
    Parameters:
    - price_data (list of float): The list of historical stock prices.
    - current_time (int): The current time step.
    - window_size (int): The number of time steps to consider for the state.
    
    Returns:
    - state (ndarray): A numpy array representing the state.
    """
    start_index = current_time - window_size + 1    
    price_block = price_data[start_index:current_time + 1] if start_index >= 0 else -start_index * [price_data[0]] + price_data[0:current_time + 1]
    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(price_block[i + 1] - price_block[i]))
    return np.array([state])

def sigmoid(n):
    """
    Applies the sigmoid function to a given value.
    
    Parameters:
    - x (float): The input value.
    
    Returns:
    - sigmoid_value (float): The result of applying the sigmoid function to x.
    """
    return 1/(1+math.exp(-n))
def train(stock_symbol, window_size, total_episodes, batch_size):
    """
    Trains a trading model on historical stock data.

    Parameters:
    - stock_symbol (str): The symbol of the stock to train on.
    - window_size (int): The number of time steps to consider for the state.
    - total_episodes (int): The total number of training episodes.
    - batch_size (int): The size of the batch for experience replay.

    Returns:
    - None
    """
    
    model_directory = "saved_models"
    losses_directory = "loss_file"
    stock_prices = stockVector(stock_symbol)
    data_length = len(stock_prices) - 1
    loss_history = []

    model = model_init(window_size, action_size)

    latest_checkpoint = find_latest_checkpoint(model_directory)
    start_episode = 0

    if latest_checkpoint:
        model = load_model(os.path.join(model_directory, f"{latest_checkpoint}.keras"))
        with open(os.path.join(losses_directory, f"losses_{latest_checkpoint}.json"), 'r') as f:
            loss_history = json.load(f)
        start_episode = latest_checkpoint + 1
        print(f"Resuming from episode {start_episode}")

    for episode in tqdm(range(start_episode, total_episodes + 1)):
        print(f"Episode {episode}/{total_episodes}")
        state = getState(stock_prices, 0, window_size + 1)
        episode_loss = 0
        total_profit = 0
        inv = []

        for step in range(data_length):
            action = act(model, state, epsilon, action_size)
            next_state = getState(stock_prices, step + 1, window_size + 1)
            reward = 0

            if action == 1:  # BUYING
                inv.append(stock_prices[step])
                print("Buy: " + priceFormat(stock_prices[step]))
            elif len(inv) > 0 and action == 2:  # SELLING
                bought_price = inv.pop(0)
                reward = max(stock_prices[step] - bought_price, 0)
                total_profit += stock_prices[step] - bought_price
                print("Sell: " + priceFormat(stock_prices[step]) + " | Profit: " + priceFormat(stock_prices[step] - bought_price))
            
            done = True if step == data_length - 1 else False
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("#################################")
                print(f"Total Profit: {priceFormat(total_profit)}")
                print("#################################")

            if len(memory) > batch_size:
                loss = exp_replay(memory, model, batch_size, gamma, epsilon, epsilon_min, epsilon_decay)
                episode_loss += loss

        average_episode_loss = episode_loss / data_length
        loss_history.append(average_episode_loss)

        # saves every 10 episodes
        if episode % 10 == 0:  
            model.save(os.path.join(model_directory, f"{episode}.keras"))
            loss_file = os.path.join(losses_directory, f"losses_{episode}.json")
            with open(loss_file, 'w') as f:
                json.dump(loss_history, f)

train("data/AMZN", 10, 2000, 32)

def load_losses_from_file(losses_file):
    with open(losses_file, 'r') as f:
        losses = json.load(f)
    return losses

episode_count = 100
losses_file = f"./loss_file/losses_{episode_count}.json"

losses = load_losses_from_file(losses_file)

# loss per episode plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses, marker='o', linestyle='-', color='blue', markersize=5, linewidth=2)
plt.yscale('log')  # Logarithmic scale for better visibility of small values
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training Loss per Episode (Log Scale)', fontsize=16)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()