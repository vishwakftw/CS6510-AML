import numpy as np
from matplotlib import pyplot as plt

results	= np.genfromtxt('results_hyp_opt.txt')[:,1:]
polynomial_res	= np.around(results[:10], 4)
gaussian_res	= np.around(results[10:-1], 4)
linear_res	= np.around(results[-1], 4)
gaussian_res	= np.array(sorted(gaussian_res.tolist(), key=lambda x: x[0]))

# Plot the Polynomial Results
plt.subplot(111)
plt.xlabel('$q$', size=20)
plt.ylabel('Accuracy (0-1 range)', size=20)
plt.title('Variation of Training and Validation Accuracies with $q$ for the Polynomial Kernel', size=20)
plt.plot(polynomial_res[:,0], polynomial_res[:,1], 'o-', label='Training Accuracy')
plt.plot(polynomial_res[:,0], polynomial_res[:,2], 'o-', label='Validation Accuracy')
plt.legend(loc='best')
plt.show()

# Plot the Gaussian Results
plt.subplot(111)
plt.xlabel('$\sigma$', size=20)
plt.ylabel('Accuracy (0-1 range)', size=20)
plt.title('Variation of Training and Validation Accuracies with $\sigma$ for the Gaussian Kernel', size=20)
plt.plot(gaussian_res[:,0], gaussian_res[:,1], 'o-', label='Training Accuracy')
plt.plot(gaussian_res[:,0], gaussian_res[:,2], 'o-', label='Validation Accuracy')
plt.legend(loc='best')
plt.show()

polynomial_best	= np.argmax(polynomial_res[:,2]) + 1
gaussian_best = (np.argmax(gaussian_res[:,2]) + 1)*0.1

fig, ax	= plt.subplots()
rect_p	= ax.bar(0.5, polynomial_res[polynomial_best - 1, 2]*100, width=0.5, color='r')
rect_g	= ax.bar(1.5, gaussian_res[int(gaussian_best*10) - 1, 2]*100, width=0.5, color='y')
rect_l	= ax.bar(2.5, linear_res[2]*100, width=0.5, color='k')
plt.annotate('{0}'.format(polynomial_res[polynomial_best - 1, 2]*100), xy=[0.5 + 0.5/2, 50], ha='center', va='center')
plt.annotate('{0}'.format(gaussian_res[int(gaussian_best*10) - 1, 2]*100), xy=[1.5 + 0.5/2, 50], ha='center', va='center')
plt.annotate('{0}'.format(linear_res[2]*100), xy=[2.5 + 0.5/2, 50], ha='center', va='center', color='w')
ax.set_title('Best validation accuracy for kernels', size=20)
ax.set_ylabel('Accuracy (in percentage)')
ax.set_xlim((0, 3.5))
ax.set_ylim((0, 100))
ax.set_xticks([0.5 + 0.5/2, 1.5 + 0.5/2, 2.5 + 0.5/2])
ax.set_xticklabels(('Polynomial', 'Gaussian', 'Linear'))
plt.show()

#@TODO: Calculate the time required to complete solving the SVM per best kernel
#@TODO: Comparison between the MKFR and individual kernels
#@TODO: Comparsion between the MKH and individual kernels and MKFR
