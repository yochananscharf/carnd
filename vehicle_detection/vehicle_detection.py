import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import time
from skimage.feature import hog
from scipy.ndimage.measurements import label





# Define a function to compute binned color features

# X_scaler = StandardScaler().fit(X)

def plot_side_by_side(images, names):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(images[0])
    ax1.set_title(names[0], fontsize=50)
    ax2.imshow(images[1], cmap='gray')
    ax2.set_title(names[1], fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def getHog():
    winsize = (64,64)
    blocksize = (16,16)
    blockstride = (8,8)
    cellsize = (8,8)
    nbins = (9)
    derivAperture = 1
    winSigma = 4
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection =0
    nlevels = 64

    return cv2.HOGDescriptor(winsize, blocksize, blockstride
                            ,cellsize,nbins,derivAperture,
                             winSigma,histogramNormType,L2HysThreshold
                             ,gammaCorrection, nlevels,)

h = getHog()


def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def get_feature(feature_image, hogD=h):
    features = []
    hist_features = color_hist(feature_image, nbins=64, bins_range=(0, 256))
    features.append(np.array(hist_features.reshape(192)))
    img_hsv = cv2.cvtColor(feature_image,cv2.COLOR_BGR2HSV)
    spatial_features = bin_spatial(img_hsv, size=(24,24))
    features.append(spatial_features)
    img = cv2.cvtColor(feature_image,cv2.COLOR_BGR2GRAY)
    hog = np.array(hogD.compute(img))
    features.append(np.array(hog).reshape(1764))
    return np.concatenate(np.array(features))#.reshape(-1,1828))

    # features = np.concatenate(spatial_features, hist_features, hog_features)
# Return list of feature vectors
    return np.concatenate(features)


def get_car_notcar_features():
    # Read in car and non-car images
    car_paths = glob.glob('../vehicles/vehicles/*/*.png')
    notcar_paths = glob.glob('../non-vehicles/non-vehicles/*/*.png')

    car_images = [cv2.imread(image) for image in car_paths]

    notcar_images = [cv2.imread(image) for image in notcar_paths]

    car_features = np.array([get_feature(img) for img in car_images])
    notcar_features = np.array([get_feature(img) for img in notcar_images])
    return car_features, notcar_features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(x_start_stop=[400,1280], y_start_stop=[360, 660],
                    xy_window=(64, 64), xy_overlap=(0.8, 0.8)):

    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    offset = 0
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx-offset, starty-offset), (endx+offset, endy+offset)))
        offset += 3
    # Return the list of windows
    return window_list


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_window = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = get_feature(test_window, h)
#         features = vd.extract_features(test_img, cspace=color_space, 
#                             spatial_size=spatial_size, hist_bins=hist_bins, 
#                             orient=orient, pix_per_cell=pix_per_cell, 
#                             cell_per_block=cell_per_block)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
#         print test_features.shape
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction > 0.99:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = (((np.min(nonzerox)-10), (np.min(nonzeroy))-10), ((np.max(nonzerox)+10), (np.max(nonzeroy)+10)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img