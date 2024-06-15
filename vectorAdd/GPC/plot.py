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

values = extract_values('incGPC.log')
x = range(1, len(values) + 1)

plt.figure(figsize=(16/2.54, 6/2.54))
plt.scatter(x, values, color='#808080', edgecolor='black', linewidth=0.4, s=25)
plt.xlabel('GPCs')
plt.ylabel('GFlops')
plt.ylim(0, 141)

# Fit with polyfit
z = np.polyfit(x, values, 2)
p = np.poly1d(z)
plt.plot(x,p(x),"k--")

# Set major ticks every 10 on x, and every 10 on y.
plt.xticks(range(1, 7, 1))
plt.yticks(range(0, 140, 10))

# Set minor ticks every 1 on x, and every 1 on y.
ax = plt.gca()
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

plt.show()

plt.savefig('incrGPC.png', dpi = 600, bbox_inches='tight')