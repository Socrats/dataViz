# coding=utf=8

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", palette="muted", color_codes=True)

results_directory = "/Users/eliasfernandez/Dropbox/Elias/Preplay/experiments_hydra/AG_DAnt_RReact_3/"

# Get all files with actions data from each of the configurations
# Count the last 1000 actions to get the distribution
# Average the results over the 10 runs
# present the results with the errors in a bar plot

dir_payoff = '//Users/eliasfernandez/PycharmProjects/PyBevo/' \
             'results/17-03-08_16:18:45_AnticipationGame_reactive_rcv_actions.npy'

actions_vector = np.load(dir_payoff)

# Set up the matplotlib figure
f, axes = plt.subplots(1, 1, figsize=(7, 7))
sns.despine(left=True)

# Plot a simple histogram with binsize determined automatically
sns.distplot(actions_vector, kde=False, color="b", bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], norm_hist=True)

plt.show()
