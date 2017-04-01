# coding=utf=8

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .paths import *

sns.set(style="white", palette="muted", color_codes=True)

# Get all files with actions data from each of the configurations
# Count the last 1000 actions to get the distribution
# Average the results over the 10 runs
# present the results with the errors in a bar plot

actions_vector = np.load(payoff_file)
actions_vector = actions_vector[0][-1000:]

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Plot a simple histogram with binsize determined automatically
sns.distplot(actions_vector, kde=False, color="b", bins=bins, norm_hist=True, ax=axes[0, 0])

# Plot a kernel density estimate and rug plot
sns.distplot(actions_vector, hist=False, rug=True, color="r", bins=bins, ax=axes[0, 1])

# Plot a filled kernel density estimate
sns.distplot(actions_vector, hist=False, color="g", kde_kws={"shade": True}, bins=bins, ax=axes[1, 0])

# Plot a historgram and kernel density estimate
sns.distplot(actions_vector, color="m", bins=bins, ax=axes[1, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()
