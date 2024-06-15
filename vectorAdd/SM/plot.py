import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def extract_values(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            values = [float(line.strip()) for line in lines]
            return values
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

# Get filename from command-line arguments
filename = 'incSM_' + sys.argv[1] + '.log' 
values = extract_values(filename)
x = range(1, len(values) + 1)

plt.figure(figsize=(16/2.54, 6/2.54))
plt.scatter(x, values, color='#808080', edgecolor='black', linewidth=0.4, s=10)
plt.xlabel('SMs')
plt.ylabel('GFlops')

# Set y-axis limit and ticks based on the maximum value in the data
y_max = max(values) + 20
plt.ylim(0, y_max)
plt.yticks(range(0, int(y_max), 15))

# Fit with polyfit
z = np.polyfit(x, values, 6)
p = np.poly1d(z)
plt.plot(x,p(x),"k--", linewidth=0.7)

# Set major ticks every 10 on x
xticks_max = int(sys.argv[1]) + 1 
plt.xticks(range(1, xticks_max, 10))

# Set minor ticks every 1 on x, and every 1 on y.
ax = plt.gca()
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

plt.show()

# Save the figure with a filename based on the input filename
plt.savefig(f'../results/incSM_{sys.argv[1]}.png', dpi = 600, bbox_inches='tight')