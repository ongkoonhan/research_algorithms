import math
import numpy as np
import cv2 as cv


class GreyORB:
    def __init__(self, **kwargs):
        self.orb = cv.ORB_create(**kwargs)   # Initiate ORB detector
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # CLAHE equalizer

    def detect_and_compute(self, img_file, equalize_hist=False, binarize=False):
        img = cv.imread(img_file, 0)
        img = self.clahe.apply(img) if equalize_hist else img
        _, img = cv.threshold(img, 180, 255, cv.THRESH_BINARY) if binarize else (None, img)
        kp, des = self.orb.detectAndCompute(img, None)   # find the keypoints and compute the descriptors with ORB
        return img, kp, des


class ColourORB:
    def __init__(self, **kwargs):
        self.orb = cv.ORB_create(**kwargs)   # Initiate ORB detector
        self.grey_world_wb = cv.xphoto.createGrayworldWB()

    def detect_and_compute(self, img_file, equalize_hist=False):
        img = cv.imread(img_file)
        img = self.grey_world_wb.balanceWhite(img) if equalize_hist else img
        kp, des = self.orb.detectAndCompute(img, None)   # find the keypoints and compute the descriptors with ORB
        return img, kp, des


class BFMatcher:
    def __init__(self):
        # self.bf = cv.BFMatcher()   # BFMatcher with default params
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)   # BFMatcher with Hamming norm for ORB, cross check replaces Lowe ratio test

    def match_and_draw(self, img1, kp1, des1, img2, kp2, des2):
        matches = self.bf.match(des1, des2)

        draw_params = dict(
            matchColor=(0, 0, 255),
            singlePointColor=(0, 255, 0),
            flags=cv.DrawMatchesFlags_DEFAULT
        )

        # matches = self.bf.knnMatch(des1, des2, k=2)
        #
        # # Need to draw only good matches, so create a mask
        # matchesMask = [[0, 0] for i in range(len(matches))]
        # # ratio test as per Lowe's paper
        # for i, (m, n) in enumerate(matches):
        #     if m.distance < 0.7 * n.distance:
        #     # if m.distance < 0.80 * n.distance:
        #         matchesMask[i] = [1, 0]
        #
        # draw_params = dict(
        #     matchColor=(0, 0, 255),
        #     singlePointColor=(0, 255, 0),
        #     matchesMask=matchesMask,
        #     flags=cv.DrawMatchesFlags_DEFAULT
        # )

        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        return img3