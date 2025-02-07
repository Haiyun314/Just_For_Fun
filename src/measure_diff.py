import matplotlib.pyplot as plt
import numpy as np

four = plt.imread('./results/heat_eq_last.png') # 50* 50
fdm = plt.imread('./results/fdm_result.png')
pinn_tf = plt.imread('./results/pinns_tf.png')
without_pinns = plt.imread('./results/without_init_pinns_tf.png')


def diff(image1, image2):
    return np.mean(np.abs(image1 - image2)), np.sqrt(np.mean(np.square(image1 - image2)))

print(f"Difference between Fourier and Finite Difference methods: L1 = {diff(four, fdm)[0]:.4f}, L2 = {diff(four, fdm)[1]:.4f}")
print(f"Difference between Fourier and PINNs: L1 = {diff(four, pinn_tf)[0]:.4f}, L2 = {diff(four, pinn_tf)[1]:.4f}")
print(f"Difference between Fourier and PINNs without initialization: L1 = {diff(four, without_pinns)[0]:.4f}, L2 = {diff(four, without_pinns)[1]:.4f}")
