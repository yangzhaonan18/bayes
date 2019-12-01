import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import style

style.use('ggplot')
alphas = [0.6, 0.15, 0.15]
betas = [0.14 * 0.14, 0.15 * 0.15, 0.1 * 0.1 ]
x = np.linspace(0, 1, 100)
f, ax = plt.subplots(2, len(alphas), sharex=True, sharey=True)
for i in range(len(alphas)):
    alpha = alphas[i]
    beta = betas[i]
    pdf = stats.beta(alpha, beta).pdf(x)
    ax[0, i].plot(x, pdf)
    ax[0, i].plot(0, 0, label='alpha={:3.2f}\nbeta={:3.2f}'.format(alpha, beta), alpha=0)
    plt.setp(ax[0, i], xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0,2,4,6,8,10])
    ax[0, i].legend(fontsize=10)
# ax[3, 0].set_xlabel('theta', fontsize=16)
# ax[0, 0].set_ylabel('pdf(theta)', fontsize=16)
plt.suptitle('Beta PDF', fontsize=12)
plt.tight_layout()
plt.show()
