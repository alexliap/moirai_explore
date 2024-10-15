import numpy as np


def moving_average(array: np.array, window_size: int):
    window_size = window_size
     
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
     
    # Loop through the array t o
    #consider every window of size 3
    while i < len(array) - window_size + 1:
     
        # Calculate the average of current window
        window_average = round(np.sum(array[i:i+window_size]) / window_size, 2)
         
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
         
        # Shift window to right by one position
        i += 1
    
    moving_averages = np.array(moving_averages)

    return moving_averages
