import matplotlib.pyplot as plt
import numpy as np

four = plt.imread('./results/heat_eq_last.png') # 50* 50
fdm = plt.imread('./results/fdm_result.png')
pinn_tf = plt.imread('./results/pinns_tf.png')
without_pinns = plt.imread('./results/without_init_pinns_tf.png')

def diff(image1, image2):
    image1 = image1/255
    image2 = image2/255
    return np.sum(np.abs(image1 - image2))

print('Difference between fourier and finite difference method:', diff(four, fdm))
print('Difference between fourier and PINNs with tensorflow:', diff(four, pinn_tf))
print('Difference between fourier and PINNs without initialization with tensorflow:', diff(four, without_pinns))
