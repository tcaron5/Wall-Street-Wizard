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
import training

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

# Choose an action
def act(model, state, epsilon=0):
    """
    Chooses an action based on the model's prediction.

    Parameters:
    - model: The trading model.
    - state: The current state of the market (input to the model).
    - epsilon (float): Exploration rate (not used in this case, hence default is 0).

    Returns:
    - int: The action chosen by the model (0 = hold, 1 = buy, 2 = sell).
    """
    options = model.predict(state)
    return np.argmax(options[0])

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
    state = training.getState(data, 0, window_size + 1)
    total_profit = 0
    inv = []
    memory = []
    
    # For plotting buy/sell signals
    buy_signals = []
    sell_signals = []

    for t in range(l):
        action = act(model, state)
        next_state = training.getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # buy
            inv.append(data[t])
            buy_signals.append((t, data[t]))  # Record buy signal
            print("Buy: " + training.priceFormat(data[t]))
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
            print(f"Total Profit: " + training.priceFormat(total_profit))
            print("################################")
            print("Total profit is:", training.priceFormat(total_profit))

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
    model_name = "./saved_models/100.keras"
    model, window_size = model_init_for_trading(model_name)
    data = stockVector(stock_name)
    print(data)

    total_profit = execute_trading_strategy(model, data, window_size)
    print("Final Total Profit:", training.priceFormat(total_profit))

# Example usage
main()