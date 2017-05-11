#!/usr/bin/python
# -*- encoding:utf8 -*-


import cv2
import numpy as np


def hist_sliding_window_points(binary_warped, roi=None, show=False):

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    if roi is None:
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    else:
        roi_left_centroid = roi[0]
        roi_right_centroid = roi[1]
        leftx_base = np.argmax(
            histogram[
                int(roi_left_centroid - margin / 2):
                int(roi_left_centroid + margin / 2)]
        ) + int(roi_left_centroid - margin / 2)
        rightx_base = np.argmax(
            histogram[
                int(roi_right_centroid - margin / 2):
                int(roi_right_centroid + margin / 2)]
        ) + int(roi_right_centroid - margin / 2)

    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if show:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        text = lambda side, fit: "%s: %.8f * y^2 + %.4f * y + %.2f" % (side, fit[0], fit[1], fit[2])

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_ppoints = np.int32([np.array(list(zip(left_fitx, ploty)))])
        cv2.polylines(out_img, left_ppoints, False, (255, 0, 0))
        right_ppoints = np.int32([np.array(list(zip(right_fitx, ploty)))])
        cv2.polylines(out_img, right_ppoints, False, (0, 0, 255))
        cv2.putText(
            out_img, text("left", left_fit),
            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200),
        )
        cv2.putText(
            out_img, text("right", right_fit),
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200),
        )

        cv2.imshow("fit", out_img)
        cv2.waitKey(-1)
        cv2.imwrite("output_images/fit.jpg", out_img)
        return []

    return [(leftx, lefty, leftx_base), (rightx, righty, rightx_base)]


YM_PER_PIX = 30.0 / 720 # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 800 # meters per pixel in x dimension


def fit(binary_warped, roi=None, show_on_birdview=False):
    pairs = hist_sliding_window_points(binary_warped, roi=roi)

    y_eval = 719 * YM_PER_PIX

    apply_poly = lambda poly: lambda y: poly[0] * y ** 2 + poly[1] * y + poly[2]

    leftx, lefty, leftx_base = pairs[0]
    left_fit = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    left_curvature = curvature(left_fit)(y_eval)
    left_position = apply_poly(left_fit)(y_eval)

    rightx, righty, rightx_base = pairs[1]
    right_fit = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)
    right_curvature = curvature(right_fit)(y_eval)
    right_position = apply_poly(right_fit)(y_eval)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]).astype(np.int32)
    left_fitx = np.int32(apply_poly(left_fit)(ploty * YM_PER_PIX) / XM_PER_PIX)
    right_fitx = np.int32(apply_poly(right_fit)(ploty * YM_PER_PIX) / XM_PER_PIX)
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.polylines(color_warp, pts_left, False, (255, 0, 0), thickness=20)
    cv2.polylines(color_warp, pts_right, False, (0, 0, 255), thickness=20)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    return left_curvature, left_position, leftx_base, right_curvature, right_position, rightx_base, color_warp


def curvature(polyfit):
    A = polyfit[0]
    B = polyfit[1]
    return lambda y: ((1 + (2 * A * y + B) ** 2) ** 1.5) / np.absolute(2 * A)
