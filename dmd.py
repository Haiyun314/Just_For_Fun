

import numpy as np
import cv2
from scipy import linalg
import scipy
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# First a useful function to convert RGB images to grayscale
def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Then, the main movie import function
def ImportMovie(fname):
    # Open video
    vidcap = cv2.VideoCapture(fname)
    FrameRate = vidcap.get(cv2.CAP_PROP_FPS)
    # Import first video frame
    success,image = vidcap.read()
    count = 0
    success = True
    movie = []
    # Import other frames until end of video
    while success:
        movie.append(rgb2gray(image))
        success, image = vidcap.read()
        count += 1
    # Convert to array
    movie = np.array(movie)
    # Display some summary information
    print("==========================================")
    print("           Video Import Summary           ")
    print("------------------------------------------")
    print("   Imported movie: ", fname)
    print(" Frames extracted: ", count)
    print("Frames per second: ", FrameRate)
    print("      data shape = ", movie.shape)
    print("==========================================")
    return movie, FrameRate

def mat2gray(image):
    out_min = np.min(image[:])
    out_max = np.max(image[:])
    idx = np.logical_and(image > out_min, image < out_max)
    image[image <= out_min] = 0
    image[image >= out_max] = 255
    image[idx] = ( 255/(out_max - out_min) ) * (image[idx] - out_min)
    return image


def CreateVideo(video_name, image_folder):
    # image_folder = "."
    # video_name = "f1slow.avi"
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = np.sort(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 60, (width, height)) 
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


# start the assignment

# transfer the frames to vectors
def vec_mo(movie):
    vec_movie = []
    for frame in movie:
        vec_movie.append(frame.reshape(shape[1]*shape[2]))
    vec_movie = np.array(vec_movie)
    vec_movie = vec_movie.transpose()  # first parameter for spatial, second parameter for temperal
    return vec_movie

path = os.getcwd()
movie, FrameRate = ImportMovie(os.path.join(path,'results/formula1.mp4'))
shape = movie.shape # time, length, width
FrameRate

vec_movie = vec_mo(movie= movie)
X_all = vec_movie[:, :100]

X_snapshot, Y_snapshot = X_all[:, :-1], X_all[:, 1:]

# the "economy" SVD
U, S, Vh  = linalg.svd(X_all, full_matrices=False) 


def conj_transpose(A):
    return np.conjugate( np.transpose( A ) )


Xr = conj_transpose(U) @ X_snapshot                   # project X snapshots
Yr = conj_transpose(U) @ Y_snapshot                   # project Y snapshots
A_POD = Yr @ linalg.pinv(Xr)

eigenvecs, eigenvals = linalg.eig(A_POD)

def high_low_frequency(A, threshold=0.03):
    eigenvals, eigenvecs = linalg.eig(A)
    omega = np.imag(np.log(eigenvals)) # project the eigenvals to frequency domain
    high_fre_mask = np.abs(omega) >= threshold
    low_fre_mask = ~high_fre_mask
    high_eig = eigenvals * high_fre_mask
    low_eig = eigenvals * low_fre_mask

    high_fre = eigenvecs @ np.diag(high_eig) @ linalg.inv(eigenvecs)
    low_fre = eigenvecs @ np.diag(low_eig) @ linalg.inv(eigenvecs)
    return high_fre, low_fre

# separate the high and low frequency
high_fre, low_fre = high_low_frequency(A_POD)

SIM_PODr_highf = np.zeros(np.shape(X_all))   
SIM_PODr_highf[:,0] = X_all[:,0]             
SIM_PODr_lowf = np.zeros(np.shape(X_all))   
SIM_PODr_lowf[:,0] = X_all[:,0]            

for k in range(np.shape(X_all)[1]-1):
    SIM_PODr_highf[:,k+1] = U @ ( high_fre @ ( conj_transpose(U) @ SIM_PODr_highf[:,k] ) ) # note the smart order of operations
    SIM_PODr_lowf[:,k+1] = U @ ( low_fre @ ( conj_transpose(U) @ SIM_PODr_lowf[:,k] ) ) # note the smart order of operations
    print(f'working on it {k/X_all.shape[1]}')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
def update(i):
    ax[0].cla()  
    ax[1].cla()
    
    ax[0].imshow(SIM_PODr_highf[:, i].reshape((shape[1], shape[2])), cmap="gray")
    ax[1].imshow(SIM_PODr_lowf[:, i].reshape((shape[1], shape[2])), cmap="gray")
    
    ax[0].set_title("High-Frequency Components")
    ax[1].set_title("Low-Frequency Components")

ani = animation.FuncAnimation(fig, update, frames=X_all.shape[1], interval=100)
output_gif_path = os.path.join(path, "results/output_animation.gif")
ani.save(output_gif_path, writer="pillow", fps=10)

