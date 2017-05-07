# -*- encoding:utf-8 -*-


import cv2
import numpy as np
import thresholds
import calibration



class LanePipeline:

    def __init__(self, calib_image_paths, test_image_path):
        self._mtx = None
        self._dist = None
        self._calib_image_paths = calib_image_paths
        self._test_image_path = test_image_path
        pass

    def _calibration(self):
        fnames, _, grays = calibration.get_images(self._calib_image_paths)
        mtx, dist = calibration.camera_calibration(grays, grays[11])
        self._mtx = mtx
        self._dist = dist
        for ith, image in enumerate(grays):
            #cv2.imshow("original", image)
            #cv2.imshow("undistort_image", self._undistort(image))
            #cv2.waitKey(-1)
            name = fnames[ith].split("/")[-1]
            cv2.imwrite("output_images/%s" % name, image)
            cv2.imwrite(
                "output_images/undistort_%s" % name, self._undistort(image))
            break

    def _undistort(self, image):
        return calibration.distortion_correction(image, self._mtx, self._dist)

    def find_lanes(self):
        self._calibration()

        test_image = cv2.imread(self._test_image_path)
        undistorted_test_image = self._undistort(test_image)
        name = self._test_image_path.split("/")[-1]
        cv2.imwrite("output_images/%s" % name, test_image)
        cv2.imwrite(
            "output_images/undistort_%s" % name, undistorted_test_image)

        binary = thresholds.combine_thresholds(undistorted_test_image)
        cv2.imshow("binary", binary.astype(np.float))
        cv2.waitKey(-1)


if __name__ == "__main__":
    pipeline = LanePipeline("camera_cal/*.jpg", "test_images/test4.jpg")
    pipeline.find_lanes()
