# -*- encoding:utf-8 -*-


import glob
import cv2
import numpy as np


def get_images(image_paths):
    images = glob.glob(image_paths)

    originals = []
    grays = []
    fnames = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fnames.append(fname)
        originals.append(img)
        grays.append(gray)

    return fnames, originals, grays


def get_corners(images, chessboard_size=(9, 6)):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]]\
        .T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for img in images:
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints


def camera_calibration(images, image):
    """
    camera_calibration, returning some intrinsic parameters.
    """
    objpoints, imgpoints = get_corners(images)

    _, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, image.shape[0:2], None, None)

    return mtx, dist


def distortion_correction(img, mtx, dist):
    """
    undistortion.
    """
    correction = cv2.undistort(img, mtx, dist, None, mtx)
    return correction


def perspective_transform(img, src=None, dst=None):
    """
    apply persp transform.
    """
    img_size = (img.shape[1], img.shape[0])
    if src is None:
        src = np.float32([
            [(img_size[0] / 2) - 50, img_size[1] / 2 + 100],
            [((img_size[0] / 6) - 20), img_size[1]],
            [(img_size[0] * 5 / 6) + 20, img_size[1]],
            [(img_size[0] / 2 + 50), img_size[1] / 2 + 100]])
    if dst is None:
        dst = np.float32([
            [(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped
