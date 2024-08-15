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

def act(model, state):
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

def model_init_for_trading(model_name):
    """
    Initializes the model for trading by loading a pre-trained model and 
    extracting the window size from the model's input layer.

    Parameters:
    - model_name (str): The file name of the saved model to load.

    Returns:
    - model: The loaded model.
    - window_size: The size of the input window used by the model.
    """
    model = load_model(model_name)
    window_size = model.layers[0].input.shape[1]
    return model, window_size


# Execute the trading strategy
def execute_trading_strategy(model, data, window_size):
    """
    Executes the trading strategy by simulating trades on historical stock data.

    Parameters:
    - model: The trained trading model.
    - data (list): List of stock prices.
    - window_size (int): The size of the input window used by the model.

    Returns:
    - float: The total profit earned from the trading strategy.
    """
    l = len(data) - 1
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    inv = []
    memory = []
    
    # For plotting buy/sell signals
    buy_signals = []
    sell_signals = []

    for t in range(l):
        action = act(model, state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # buy
            inv.append(data[t])
            buy_signals.append((t, data[t]))  # Record buy signal
            print("Buy: " + priceFormat(data[t]))
        elif action == 2 and len(inv) > 0:  # sell
            bought_price = inv.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            sell_signals.append((t, data[t]))  # Record sell signal
            print("Sell: " + priceFormat(data[t]) + " | Profit: " + priceFormat(data[t] - bought_price))

        done = True if t == l - 1 else False
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("################################")
            print(f"Total Profit: " + priceFormat(total_profit))
            print("################################")
            print("Total profit is:", priceFormat(total_profit))

    # Plotting the stock price with buy and sell signals
    plt.figure(figsize=(15, 5))
    plt.plot(data, label='Stock Price')
    buy_points = [p[1] for p in buy_signals]
    buy_times = [p[0] for p in buy_signals]
    sell_points = [p[1] for p in sell_signals]
    sell_times = [p[0] for p in sell_signals]

    plt.scatter(buy_times, buy_points, marker='^', color='green', label='Buy Signal', alpha=1)
    plt.scatter(sell_times, sell_points, marker='v', color='red', label='Sell Signal', alpha=1)
    plt.title('Stock Price with Buy and Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return total_profit

# Initializes the trading model and executes the trading strategy on a specified stock.
def main():
    stock_name = "data/AMZN"
    model_name = "./saved_models/50.keras"
    model, window_size = model_init_for_trading(model_name)
    data = stockVector(stock_name)
    print(data)

    total_profit = execute_trading_strategy(model, data, window_size)
    print("Final Total Profit:", priceFormat(total_profit))

# Example usage
main()