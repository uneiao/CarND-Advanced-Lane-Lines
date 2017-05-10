# -*- encoding:utf-8 -*-


import cv2
import numpy as np
import thresholds
import calibration
import line_fit
from moviepy.editor import VideoFileClip


class LanePipeline:

    def __init__(self, calib_image_paths, test_image_path):
        self._mtx = None
        self._dist = None
        self._calib_image_paths = calib_image_paths
        self._test_image_path = test_image_path
        self._is_save_image = True

    def _calibration(self):
        _pickle_fname = "mtx_dist.pickled"
        import os.path
        import pickle
        if os.path.isfile(_pickle_fname):
            res = pickle.load(open(_pickle_fname, "rb"))
            self._mtx = res["mtx"]
            self._dist = res["dist"]
            return

        fnames, _, grays = calibration.get_images(self._calib_image_paths)
        mtx, dist = calibration.camera_calibration(grays, grays[11])
        self._mtx = mtx
        self._dist = dist
        res = {
            "mtx": mtx,
            "dist": dist,
        }
        pickle.dump(res, open(_pickle_fname, "wb"))
        for ith, image in enumerate(grays):
            #cv2.imshow("original", image)
            #cv2.imshow("undistort_image", self._undistort(image))
            #cv2.waitKey(-1)
            name = fnames[ith].split("/")[-1]
            self.save_image("output_images/%s" % name, image)
            self.save_image(
                "output_images/undistort_%s" % name, self._undistort(image))
            break

    def _undistort(self, image):
        return calibration.distortion_correction(image, self._mtx, self._dist)

    def save_image(self, name, image):
        if self._is_save_image:
            cv2.imwrite(name, image)

    def get_bird_view_params(self, image, show=False):
        src = np.float32([(295, 680), (595, 450), (700, 450), (1150, 680)])

        if show:
            outline = cv2.polylines(
                np.copy(image), np.int32([src]),
                isClosed=True, color=(0, 0, 255), thickness=1)
            self.save_image("output_images/pick_polygon_persp.jpg", outline)
            cv2.imshow("outline", outline)
            cv2.waitKey(-1)

        dst = np.float32([(250, 710), (250, 0), (1050, 0), (1050, 710)])

        if show:
            birdview_image = calibration.perspective_transform(
                outline, src, dst)
            self.save_image(
                "output_images/pick_polygon_birdview.jpg", birdview_image)
            cv2.imshow("birdview", birdview_image)
            cv2.waitKey(-1)

        return src, dst

    def find_lanes(self):
        self._calibration()
        import glob
        for image_path in glob.glob("test_images/*.jpg"):
            test_image = cv2.imread(image_path)
            image = self.find_lane_on_image(test_image)
            cv2.imshow(image_path, image)
            cv2.waitKey(-1)

    def find_lane_on_image(self, image):
        undistorted_image = self._undistort(image)
        #name = self._test_image_path.split("/")[-1]
        #self.save_image("output_images/%s" % name, test_image)
        #self.save_image(
        #    "output_images/undistort_%s" % name, undistorted_image)

        src, dst = self.get_bird_view_params(undistorted_image, False)

        thresholded_image = thresholds.combine_thresholds(undistorted_image)
        self.save_image("output_images/thresholded_image.jpg", thresholded_image)
        #cv2.imshow("thresholded", thresholded_image.astype(np.float))
        #cv2.waitKey(-1)

        birdview_image, _ = calibration.perspective_transform(
            thresholded_image, src, dst)
        #cv2.imshow("birdview", birdview_image.astype(np.float32))
        #cv2.waitKey(-1)

        left_curvature, left_position, left_hist_base, \
            right_curvature, right_position, right_hist_base, \
            color_warp = line_fit.fit(birdview_image)

        result = self.draw_lane(undistorted_image, color_warp)
        return result

    def draw_lane(self, undist, color_warp):
        src, dst = self.get_bird_view_params(None, False)
        color_warp_inv, _ = calibration.perspective_transform(
            color_warp, dst, src)

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, color_warp_inv, 0.3, 0)
        return result

    def process_video(self, video_file_path):
        self._calibration()
        def process_image(image):
            return self.find_lane_on_image(image)
        video_output = "%s_output.mp4" % video_file_path.split(".")[0]
        clip1 = VideoFileClip(video_file_path)
        _clip = clip1.fl_image(process_image)
        _clip.write_videofile(video_output, audio=False)


if __name__ == "__main__":
    pipeline = LanePipeline("camera_cal/*.jpg", "test_images/test4.jpg")
    pipeline.find_lanes()
    #pipeline.process_video("project_video.mp4")
