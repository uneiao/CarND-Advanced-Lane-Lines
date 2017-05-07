# -*- encoding:utf-8 -*-


import numpy as np
import cv2


def combine_thresholds(img):

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[
            (scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_output

    def mag_thresh(sobelx, sobely, mag_thresh=(0, 255)):
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output

    def dir_threshold(sobelx, sobely, thresh=(0, np.pi/2)):
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    def hls_threshold(hls, sx_thresh, s_thresh):
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[
            (scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
        return color_binary, sxbinary, s_binary

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(
        gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(
        gray, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    #mag_binary = mag_thresh(sobelx, sobely, mag_thresh=(30, 100))
    #dir_binary = dir_threshold(sobelx, sobely, thresh=(0.7, 1.3))
    #_, l_binary, s_binary = hls_threshold(hls, (20, 100), (170, 250))
    gradx_s_channel = abs_sobel_thresh(s_channel, thresh=(20, 150))

    # Apply thresholds
    combined_binary = np.zeros_like(gradx)
    combined_binary[
        ((grady == 1) & (gradx == 1) | (gradx_s_channel == 1))
    ] = 1

    return combined_binary
