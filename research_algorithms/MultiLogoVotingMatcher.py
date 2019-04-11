import math
import numpy as np
import cv2 as cv

from project.ScanLinePolygonFill import ScanLinePolygonFill


# Requires ScanLinePolygonFill from https://github.com/ongkoonhan/python_algorithms/blob/master/python_algorithms/ScanLinePolygonFill.py

# Kurzejamski, G., Zawistowski, J., & Sarwas, G. (2016). Robust method of vote aggregation and proposition verification for invariant local features. arXiv preprint arXiv:1601.00781.
# https://arxiv.org/pdf/1601.00781.pdf


class MultiLogoVotingMatcher:

    def __init__(self, img_q, kp_q, des_q, centre_pt_img_q=None):
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)  # BFMatcher with Hamming norm for ORB, matches all features
        self.img_q = img_q
        self.kp_q = kp_q
        self.des_q = des_q
        self.centre_pt_img_q = centre_pt_img_q

    def match_and_create_vote_space_vote_image(self, img_t, kp_t, des_t):
        # Get matches and distance threshold
        matches = self.bf.match(des_t, self.des_q)  # Match train to query

        distances = [m.distance for m in matches]
        dist_threshold = (min(distances) + max(distances)) * 0.5

        # Calculate centre points
        centre_pt_img_q = np.array((self.img_q.shape[1] / 2.0, self.img_q.shape[0] / 2.0)) if self.centre_pt_img_q is None else np.array(self.centre_pt_img_q)

        # Run voting procedure
        vote_space, vote_img = {}, np.zeros(img_t.shape, np.float)
        for m, d in zip(matches, distances):
            if d > dist_threshold:
                continue
            else:
                kp_t_pt_idx, kp_q_pt_idx = m.queryIdx, m.trainIdx
                kp_t_pt, kp_q_pt = kp_t[kp_t_pt_idx], self.kp_q[kp_q_pt_idx]
                angle_diff = (kp_t_pt.angle - kp_q_pt.angle + 360) % 360   # diff between kp_t and kp_q mod 360
                scale_diff = kp_t_pt.size / kp_q_pt.size   # diff between kp_t and kp_q

                rel_disp_from_centre_q = np.subtract(centre_pt_img_q, np.array(kp_q_pt.pt))  # query image
                rotation_matx = cv.getRotationMatrix2D((0,0), angle_diff, scale_diff)
                rotation_matx = rotation_matx[:, :2]   # change 2x3 to 2x2
                rel_disp_from_centre_t = np.matmul(rotation_matx, rel_disp_from_centre_q)

                rel_centre_pt_t = np.add(np.array(kp_t_pt.pt), rel_disp_from_centre_t)
                x, y = tuple(np.ravel(rel_centre_pt_t))
                x, y = int(round(x)), int(round(y))
                if x < 0 or x >= img_t.shape[1]:   # out of bounds
                    continue
                if y < 0 or y >= img_t.shape[0]:   # out of bounds
                    continue
                adj_value = 1 - (d/dist_threshold)**2

                vote_img[y][x] += adj_value
                vote_pt = (x,y)
                vote_param = {
                    "kp_t_pt": kp_t_pt,
                    "kp_q_pt": kp_q_pt,
                    "kp_t_pt_idx": kp_t_pt_idx,
                    "kp_q_pt_idx": kp_q_pt_idx,
                    "angle_diff": angle_diff,
                    "scale_diff": scale_diff,
                    "adj_value": adj_value,
                }
                if vote_pt in vote_space:
                    vote_space[vote_pt].append(vote_param)
                else:
                    vote_space[vote_pt] = [vote_param]

        vote_img = vote_img * (255 / np.max(vote_img))  # normalize
        vote_img = vote_img.astype(np.uint8)
        # vote_img = vote_img.astype(np.float32)
        return vote_space, vote_img

    def find_propositions(self, vote_img, maxCorners=10, qualityLevel=0.01, minDistance=10):
        _, buf = cv.imencode(".png", vote_img)   # Encoding and decoding to fix some data type issue with goodFeaturesToTrack()
        vote_img = cv.imdecode(buf, cv.IMREAD_GRAYSCALE)
        corners = cv.goodFeaturesToTrack(vote_img, maxCorners, qualityLevel, minDistance)
        propositions = [] if corners is None else np.int0(corners)
        propositions = [tuple(np.ravel(pt)) for pt in propositions]
        return propositions

    def detect_objects_from_votes(self, vote_space, propositions, local_area_size=(21,21)):
        def roi_search_dim_ranges(centre_pt, local_area_size):
            dim_ranges = []
            for i in range(len(local_area_size)):
                dim_l = centre_pt[i] - (local_area_size[i] - 1) // 2
                dim_u = centre_pt[i] + (local_area_size[i] - 1) // 2 + 1
                dim_ranges.append(np.arange(dim_l, dim_u))
            return dim_ranges

        def unique_filtering(local_area_pts):
            filtered_pts = {}
            for vote_param in local_area_pts:
                kp_q_pt_idx = vote_param["kp_q_pt_idx"]
                if kp_q_pt_idx in filtered_pts:
                    if vote_param["adj_value"] <= filtered_pts[kp_q_pt_idx]["adj_value"]:
                        continue
                filtered_pts[kp_q_pt_idx] = vote_param
            return list(filtered_pts.values())

        def cascade_filtering(local_area_pts, vote_count_thres=3, adj_sum_thres=1.0):
            # (1) Vote count thresholding
            if len(local_area_pts) < vote_count_thres:
                return False

            # (2) Adjacency sum thresholding
            if sum([vote_param["adj_value"] for vote_param in local_area_pts]) < adj_sum_thres:
                return False

            # (3) Scale variance thresholding
            # (4) Rotation variance thresholding
            # (5) Feature points binary test
            # (6) Normalized global luminance cross correlation
            return True

        def estimate_rectangle_vertices(img_orig, new_centre_pt, angle_diff_estimate, scale_diff_estimate):
            x, y = img_orig.shape[1], img_orig.shape[0]
            rect_pts = [(0,0), (0,y), (x,y), (x,0)]
            rect_pts = np.asarray([(a-x/2.0, b-y/2.0) for a,b in rect_pts])

            # Rotate and scale, then translate (preserve the translation)
            # Rotate and Scale
            rotation_matx = cv.getRotationMatrix2D((0,0), angle_diff_estimate, scale_diff_estimate)
            rotation_matx = rotation_matx[:, :2]  # change 2x3 to 2x2
            rect_pts = np.matmul(rect_pts, rotation_matx)

            # Translate
            displacement = np.asarray([new_centre_pt] * 4)   # one for each vertex
            rect_pts = np.add(rect_pts, displacement)

            rect_pts = rect_pts.astype(int)
            rect_pts = [tuple(np.ravel(pt)) for pt in rect_pts]
            return rect_pts

        ### Proc start

        # Sort propositions
        proposition_adj_values = []
        for pt in propositions:
            if pt not in vote_space:
                continue
            sum_adj_values = sum([vote_param["adj_value"] for vote_param in vote_space[pt]])
            proposition_adj_values.append((pt, sum_adj_values,))
        proposition_adj_values.sort(key=lambda x: x[1], reverse=True)   # Descending

        estimated_object_rectangles = []
        for (pt, sum_adj_values) in proposition_adj_values:
            ### First Pass Filtering
            # Gather votes in local area
            local_area_pts = []
            dim_ranges = roi_search_dim_ranges(pt, local_area_size)
            for x in dim_ranges[0]:
                for y in dim_ranges[1]:
                    local_area_pt = (x,y)
                    local_area_pts.extend(vote_space[local_area_pt]) if local_area_pt in vote_space else None   # flatted lists of pts
            if len(local_area_pts) == 0:
                continue

            # Unique filtering
            # Keep pts with the highest adj_value for each corresponding keypoint in the query image
            local_area_pts = unique_filtering(local_area_pts)   # Update with filtered points

            # Cascade filtering
            if not cascade_filtering(local_area_pts, vote_count_thres=3, adj_sum_thres=1.0):
                continue

            ### Second Pass Filtering
            # Estimate object size
            scale_diff_estimate = sum([math.log(vote_param["scale_diff"]) for vote_param in local_area_pts]) / len(local_area_pts)
            scale_diff_estimate = math.exp(scale_diff_estimate)   # Geometric mean using arithmetic mean of log values
            angle_diff_estimate = math.log(sum([vote_param["angle_diff"] for vote_param in local_area_pts]), 360) - math.log(len(local_area_pts), 360)   # Mean, division mod 360 done using log base 360
            angle_diff_estimate = math.pow(angle_diff_estimate, 360) % 360
            # TODO: Replace mean angle estimate using circular statistical measuere for better accuracy

            # Create rotated rectangle
            rect_pts = estimate_rectangle_vertices(self.img_q, pt, angle_diff_estimate, scale_diff_estimate)

            local_area_pts = []
            to_remove = []
            for (x,y) in ScanLinePolygonFill(rect_pts).iterate_fill_polygon_pts():
                local_area_pt = (x,y)
                local_area_pts.extend(vote_space[local_area_pt]) if local_area_pt in vote_space else None  # flatted lists of pts
                to_remove.append(local_area_pt)

            # Unique filtering
            # Keep pts with the highest adj_value for each corresponding keypoint in the query image
            local_area_pts = unique_filtering(local_area_pts)  # Update with filtered points

            # Cascade filtering
            if not cascade_filtering(local_area_pts, vote_count_thres=6, adj_sum_thres=2.0):
                continue

            # Re-Estimate object size
            scale_diff_estimate = sum([math.log(vote_param["scale_diff"]) for vote_param in local_area_pts]) / len(local_area_pts)
            scale_diff_estimate = math.exp(scale_diff_estimate)  # Geometric mean using arithmetic mean of log values
            angle_diff_estimate = math.log(sum([vote_param["angle_diff"] for vote_param in local_area_pts]), 360) - math.log(len(local_area_pts), 360)  # Mean, division mod 360 done using log base 360
            angle_diff_estimate = math.pow(angle_diff_estimate, 360) % 360
            # TODO: Replace mean angle estimate using circular statistical measuere for better accuracy

            # Create rotated rectangle
            rect_pts = estimate_rectangle_vertices(self.img_q, pt, angle_diff_estimate, scale_diff_estimate)
            estimated_object_rectangles.append({
                "rect_pts": rect_pts,
                "scale_diff_estimate": scale_diff_estimate,
                "angle_diff_estimate": angle_diff_estimate,
            })

            # Remove pts in detected-object region
            for pt in to_remove:
                vote_space.pop(pt, None)

        return estimated_object_rectangles
































