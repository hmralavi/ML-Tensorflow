import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps

# set cv2 random seed
cv2.setRNGSeed(0)

# Load image
image = cv2.imread(r'..\datasets\apple2orange\testA\n07740461_1470.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR to RGB
height, width, channels = image.shape

# Reshape image to a list of pixels, each pixel is a feature vector with (B, G, R, x, y)
data = image.reshape((-1, channels))
coordinates = np.column_stack((np.repeat(np.arange(height), width), np.tile(np.arange(width), height)))
data_with_coordinates = np.hstack((data, coordinates))

# Convert to float32 for kmeans
data_with_coordinates = np.float32(data_with_coordinates)
data = np.float32(data)

# Specify criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.2)
k = 4  # Number of clusters
_, labels, centers = cv2.kmeans(data_with_coordinates, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now 'labels' contains the cluster assignments for each pixel
output_img = np.zeros_like(data)
cmap = colormaps['viridis'].resampled(8)(np.linspace(0,1,k))*255 # creating different colors for each label
for c in range(k): # a loop that goes over each label and assigns a unique color to the corresponding pixels
    output_img[labels.flatten()==c, :] = cmap[c,:3]

output_img = output_img.reshape(image.shape).astype(np.uint8)

# plotting
_, axs = plt.subplots(1,2)
axs[0].imshow(image)
axs[1].imshow(output_img)
plt.show()