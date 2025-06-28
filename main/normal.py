# coding=utf-8
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 参数设置
mu = 0.1
sigma = 0.07
sample_size = 10000

samples = [abs(random.gauss(mu, sigma)) for _ in range(sample_size)]
pf = pd.DataFrame(samples, columns=['value'])
print(pf.describe())

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
sns.histplot(samples, kde=True, bins=50, stat="density", color="skyblue", ax=ax[0])
ax[0].set_xlim(mu-3*sigma, mu+3*sigma)
ax[0].set_ylim(-1, 10)

ax[1].set_xlim(mu-3*sigma, mu+3*sigma)
x = np.linspace(-1, 1, 10000)
p = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
ax[1].plot(x, p, linewidth=2, label=f'N ~ ({mu},{sigma})')
ax[1].set_ylim(-1, 10)

ax[0].set_title('Sample Figure')
ax[1].set_title('Simulation Figure')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join("runs", "normal.png"))
plt.show()    
