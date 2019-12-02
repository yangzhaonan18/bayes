from scipy import stats

import numpy as np

print(np.random.beta(1.5, 5))

sample = np.random.beta(1.5, 5)






# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.beta.html
# scipy.stats.beta
from scipy.stats import beta
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

a, b = 1.5, 5
# mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')

# rv = beta(a, b)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')


# vals = beta.ppf([0.001, 0.5, 0.999], a, b)
# np.allclose([0.001, 0.5, 0.999], beta.cdf(vals, a, b))


r = beta.rvs(a, b, size=1000)


ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
# plt.show()






if __name__ == '__main__':
	print(np.random.beta(1.5, 5))