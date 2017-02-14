import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2, sys
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from functions import *
from skimage.feature import hog

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

import fnmatch
import os

USE_IMAGE_HOG = True
PROCESS_IMAGE = False
VIDEOFILE="../CarND-Vehicle-Detection/project_video.mp4"
SKIP_COUNT=0
SCALER_MODEL_FILE = "Scaler_Car_Classifier.pkl"
SVC_MODEL_FILE = "SVC_Car_Classifier.pkl"
FORCE_TRAIN_CLASSIFIER = False
HEATMAP_THRESHOLD = 13
HEAT_SCALING = 0.9
ENABLE_HEATMAP_HISTORY = True
ROI_Y1, ROI_Y2 = 400, 660
IMAGE_SCALING = 1.3

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

def walk_dirs(dirname):
    for root, dirnames, filenames in os.walk(dirname):
        for filename in fnmatch.filter(filenames, '*.png'):
            image_fnames.append(os.path.join(root, filename))
    return image_fnames

image_fnames = []
def get_image_names():
    global image_fnames

    image_fnames = walk_dirs('./data')
    if len(image_fnames) == 0:
        download_data()
        image_fnames = walk_dirs('./data')

def download_data():
    print ("*** No images found in folder ../datai. Downloading data. Please execute the script again ***");
    if not os.path.isdir("./CarND-Advanced-Lane-Lines"):
        os.system("git clone https://github.com/udacity/CarND-Advanced-Lane-Lines.git");
    if not os.path.isfile("./vehicles.zip"):
        os.system("wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip");
    if not os.path.isfile("./non-vehicles.zip"):
        os.system("wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip");
    if not os.path.isdir("./data"):
        os.system("mkdir data")
    if not os.path.isdir("./data/vehicles"):
        os.system("unzip vehicles.zip -d data")
        os.system("rm -rf data/__M*")
    if not os.path.isdir("./data/non-vehicles"):
        os.system("unzip non-vehicles.zip -d data")
        os.system("rm -rf data/__M*")

# Divide up into cars and notcars
cars = []
notcars = []
get_image_names()
for image_fname in image_fnames:
    if 'non-vehicles' in image_fname:
        notcars.append(image_fname)
    else:
        cars.append(image_fname)

# return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    scale = 0.0
    width_new = xy_window[0]
    height_new = xy_window[1]

    #for ys in range(ny_windows):
    while True:
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = int(startx + width_new)
            #starty = ys*ny_pix_per_step + y_start_stop[0]
            starty = y_start_stop[0]
            endy = int(starty + height_new)

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

        width_new  += nx_pix_per_step * (1-xy_overlap[0])
        height_new += ny_pix_per_step * (1-xy_overlap[1])
        scale += 1

        if height_new >= yspan:
            break;

    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def single_feat_vec(img_in, hog0, hog1, hog2, window,
                    spatial_feat = True, hist_feat = True, hog_feat = True):
    #1) Define an empty list to receive features
    img_features = []

    #2) cut out region specified by input window
    wx1 = window[0][0]
    wy1 = window[0][1]
    wx2 = window[1][0]
    wy2 = window[1][1]
    img = img_in[wy1:wy2, wx1:wx2, :]

    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(img, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(img, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    hogf = []
    if hog_feat == True:
        whx1 = wx1/pix_per_cell
        why1 = wy1/pix_per_cell
        whw = (wx2-wx1)/pix_per_cell-1
        whh = (wy2-wy1)/pix_per_cell-1

        hogf0 = hog0[why1:why1+whh, whx1:whx1+whw].ravel()
        hogf1 = hog1[why1:why1+whh, whx1:whx1+whw].ravel()
        hogf2 = hog2[why1:why1+whh, whx1:whx1+whw].ravel()

        hogf = np.hstack((hogf0, hogf1, hogf2))

        #8) Append features to list
        img_features.append(hogf)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    else: feature_image = np.copy(img)

    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)

        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    #8) Return windows for positive detections
    return on_windows

def hstack_img(src1, src2):
    src1 = np.array(src1)
    src2 = np.array(src2)

    if(len(src1.shape) == 2):
        src1 = scale_img_vals(src1, 255)
    if(len(src2.shape) == 2):
        src2 = scale_img_vals(src2, 255)

    dst = np.concatenate((src1, src2), axis=1)
    dst = cv2.resize(dst, src1.shape[:2][::-1])
    return dst

from matplotlib import pyplot as plt
def visualize_car_classifier():
    image = mpimg.imread(notcars[14])
    img = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    car_hog_img = np.zeros_like(img)
    for i in range(img.shape[2]):
        car_features, car_hog_img[:,:,i] = get_hog_features(img[:,:,i], orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                            vis=True, feature_vec=True)

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Input image")
    plt.subplot(1,2,2)
    plt.imshow(car_hog_img[:,:,0])
    plt.title("HOG features")
    plt.show()
    cv2.waitKey(0)
    return

#sample_size = 1500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]
def train_car_classifier():
    car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell', cell_per_block,'cells per block', spatial_size, 'spatial features',
          hist_bins, 'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    save_trained_classifier(svc, X_scaler)
    return svc, X_scaler

from sklearn.externals import joblib
def save_trained_classifier(svc, X_scaler):
    joblib.dump(svc, SVC_MODEL_FILE)
    joblib.dump(X_scaler, SCALER_MODEL_FILE)

def restore_saved_classifier():
    print ("Restoring saved model  " + SVC_MODEL_FILE);
    svc = joblib.load(SVC_MODEL_FILE)
    print ("Restoring saved model  " + SCALER_MODEL_FILE);
    X_scaler = joblib.load(SCALER_MODEL_FILE)
    return svc, X_scaler

import os.path
def check_for_saved_SVC_classifier():
    return os.path.isfile(SVC_MODEL_FILE)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1*HEAT_SCALING

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    y2t = ROI_Y1
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1];
        if y2 > y2t:
            y2t = y2
        w, h = x2-x1, y2-y1
        if (w >= h):
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img, y2t

def visualize_heat_map(image, heatmap):
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Input image")
    plt.subplot(1,2,2)
    plt.imshow(heatmap)
    plt.title("Heat map")
    plt.show()
    cv2.waitKey(0)

from scipy.ndimage.measurements import label
gHeatmap = None
heat_count = 0
HEAT_HISTORY_PLANES = 1
heat_threshold = HEATMAP_THRESHOLD
if ENABLE_HEATMAP_HISTORY:
    HEAT_HISTORY_PLANES = 5

def process_heat_map(image, hot_windows):
    global heat_count, gHeatmap, heat_threshold, HEAT_SCALING
    if gHeatmap == None:
        gHeatmap = np.zeros((image.shape[0], image.shape[1], HEAT_HISTORY_PLANES))#[:,:,0]).astype(np.float)
    
    heat_count += 1
    heat_count %= HEAT_HISTORY_PLANES
    gHeatmap[:,:,heat_count] = np.zeros((image.shape[0], image.shape[1]))#[:,:,0]).astype(np.float)
    gHeatmap[:,:,heat_count] = add_heat(gHeatmap[:,:,heat_count], hot_windows)
    heatmap = np.sum(gHeatmap, axis=2)

    #if np.max(heatmap) >= 30:
        #visualize_heat_map(image, heatmap)

    apply_threshold(heatmap, heat_threshold)
    labels = label(heatmap)

    image, y2t = draw_labeled_bboxes(image, labels)
    if labels[1] > 0:
        heat_threshold = HEATMAP_THRESHOLD - 4
        HEAT_SCALING = 1.0
    else:
        heat_threshold = HEATMAP_THRESHOLD
        HEAT_SCALING = 1.8
    global IMAGE_SCALING
    IMAGE_SCALING = 1.0 + (y2t + 40 - ROI_Y1)/(ROI_Y2-ROI_Y1) #scaling should be between 1.5 and 1
    return image

def debug_windows(image, windows):
    temp_image = image.copy()
    for i in range(len(windows)):
        print i, len(windows), windows[i-1:i]
        temp_image = draw_boxes(temp_image, windows[i-1:i], color=((i%3)*85,(i%5)*51,(i%10)*25), thick=2)
        cv2.imshow("Output bbox", temp_image)
    cv2.imwrite("debug/sliding_window.jpg", temp_image)
    cv2.waitKey(0)
    exit(0)
    return image

def detect_vehicles_scaled_image(image):
    draw_image = np.copy(image)
    image = image.astype(np.float32)/255
    image = image[ROI_Y1:ROI_Y2, :, :]

    hog_features = []
    window_shift_cells = 1
    shift = window_shift_cells * pix_per_cell
    window_w, window_h = 64, 64
    scale = IMAGE_SCALING
    image = cv2.resize(image, (int(image.shape[1]/scale), int(image.shape[0]/scale)))

    fimage = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    hog0 = get_hog_features(fimage[:,:,0], orient, pix_per_cell, cell_per_block,
                            vis=False, feature_vec=False)
    hog1 = get_hog_features(fimage[:,:,1], orient, pix_per_cell, cell_per_block,
                            vis=False, feature_vec=False)
    hog2 = get_hog_features(fimage[:,:,2], orient, pix_per_cell, cell_per_block,
                            vis=False, feature_vec=False)
    windows = []
    on_windows = []
    for y in range(0, image.shape[0]-window_h, shift):
        for x in range(0, image.shape[1]-window_w, shift):
            window = ((x,y), (x+window_w,y+window_h))
            windows.append(((int(x*scale),int((y*scale)+ROI_Y1)), (int((x+window_w)*scale),int((y+window_h)*scale+ROI_Y1))))
            features = single_feat_vec(fimage, hog0, hog1, hog2, window)
            #5) Scale extracted features to be fed to classifier
            test_features = X_scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = svc.predict(test_features)

            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                window = ((int(x*scale),int((y*scale)+ROI_Y1)), (int((x+window_w)*scale),int((y+window_h)*scale+ROI_Y1)))
                on_windows.append(window)

    #debug_windows(draw_image, windows)
    window_img = draw_boxes(draw_image, on_windows, color=(0, 0, 255), thick=6)
    hp_image = process_heat_map(draw_image, on_windows)
    cv2.imshow("Detected vehicles", hstack_img(window_img, hp_image))
    return hp_image

def detect_vehicles_scaled_windows(image):
    draw_image = np.copy(image)

    # Input image is jpg range(0-255) training happened with png range(0.0-1.0) scale 0 to 255
    image = image.astype(np.float32)/255

    y_start_stop = [ROI_Y1, ROI_Y2] # Min and max in y to search in slide_window()
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(64, 64), xy_overlap=(0.45, 0.45))

    #debug_windows(draw_image, windows)
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    hp_image = process_heat_map(draw_image, hot_windows)

    cv2.imshow("Output combined", hstack_img(window_img, hp_image));

    return hp_image

def process_image(image):
    if USE_IMAGE_HOG:
        return detect_vehicles_scaled_image(image)
    else:
        return detect_vehicles_scaled_windows(image)

if((check_for_saved_SVC_classifier() == False)  or FORCE_TRAIN_CLASSIFIER):
    svc, X_scaler = train_car_classifier()
else:
    #visualize_car_classifier()
    svc, X_scaler = restore_saved_classifier()

if PROCESS_IMAGE:
    image = cv2.imread('CarND-Advanced-Lane-Lines/test_images/test1.jpg')
    output_img = process_image(image)
    cv2.imshow("Output", output_img)
    cv2.waitKey(0)
else:
    out = None
    fourcc = cv2.cv.CV_FOURCC(*'XVID')

    frame_count = 0
    cap = cv2.VideoCapture(VIDEOFILE)
    while True:
        ret, image = cap.read()
        if ret == False:
            break

        IMAGE_HEIGHT, IMAGE_WIDTH, c = image.shape

        if out is None:
            out = cv2.VideoWriter('output.avi',fourcc, 25.0, (IMAGE_WIDTH,IMAGE_HEIGHT))

        if ret == True:
            frame_count +=  1
            if(frame_count < SKIP_COUNT):
                continue
            print frame_count

            output_img = process_image(image)
            out.write(output_img)
            #cv2.imwrite("debug/detected_cars.png", output_img)

            c = cv2.waitKey(31) & 0x7F
            if c == 'q' or c == 27:
                exit(0)
