import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np
import seaborn as sns

alphas = np.load("./data_npy/alphas.npy")
betas = np.load("./data_npy/betas.npy")

samples_matrix = np.load("samples_matrix.npy")









def show_21():

	plt.figure(figsize=(15, 7))
	for i in range(21):
	    plt.subplot(7, 3, i+1)
	    sns.distplot(samples_matrix[:, i], hist=False,  color='red')
	 


	    # plt.hist(samples_matrix[:, i], bins=100, color='red',histtype='stepfilled',alpha=0.75)
	    a, b = float(alphas[i]), float(betas[i])
	    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
	    plt.plot(x, beta.pdf(x, a, b), 'b-', lw=2, alpha=0.6, label='beta pdf')
	    plt.title('%s' % i)

	plt.savefig('out.png')
	# plt.show()


def show_one():

	names = ["T00","T01","T02","T10","T11","T12","T20","T21","T22",
			 "V00","V01","V02","V10","V11","V12","V20","V21","V22",
			"S0","S1","S2"]
	for  i in range(21):
		name = names[i]
		plt.figure()
		# i = 1
		# plt.subplot(7, 3, i+1)
		sns.distplot(samples_matrix[:, i], hist=False, color='red',label="Marginal posterior of " + name)


		# plt.hist(samples_matrix[:, i], bins=100, color='red',histtype='stepfilled',alpha=0.75)
		a, b = float(alphas[i]), float(betas[i])
		x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
		plt.plot(x, beta.pdf(x, a, b), 'b--', lw=2, alpha=0.6, label='Priori of ' + name)
		plt.legend()
		plt.title('%s' % i)
		plt.show()


if __name__ == "__main__":


	show_one()

	import pandas as pd
	df = pd.DataFrame(samples_matrix)
	 
	dfData = df.corr() 
	plt.subplots(figsize=(9, 9)) # ÉèÖÃ»­Ãæ´óÐ¡
	# sns.heatmap(dfData, vmax=128, annot=True, square=True, cmap="gist_gray")
	sns.heatmap(dfData, vmax=0.1, square=True, cmap="gist_gray")
	# plt.savefig('./BluesStateRelation.png')
	plt.title('Posterior Parameters')
	plt.show() 