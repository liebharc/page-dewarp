import sys
from typing import Tuple
import cv2
import numpy as np


def image_threshold(img: np.ndarray) -> np.ndarray:
    dst = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    mask = cv2.adaptiveThreshold(
        src=gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=101,
        C=21,
    )
    return cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=5).astype(np.uint8)


def get_large_shapes(img):
    blur = cv2.medianBlur(img, 9)
    thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    return close


def blob_mean_and_tangent(contour):
    """
    Construct blob image's covariance matrix from second order central moments
    (i.e. dividing them by the 0-order 'area moment' to make them translationally
    invariant), from the eigenvectors of which the blob orientation can be
    extracted (they are its principle components).
    """
    moments = cv2.moments(contour)
    area = moments["m00"]
    if area == 0:
        area = 1
    mean_x = moments["m10"] / area
    mean_y = moments["m01"] / area
    covariance_matrix = np.divide(
        [[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]], area
    )
    _, svd_u, _ = cv2.SVDecomp(covariance_matrix)
    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()
    return center, tangent


class Contour:
    def __init__(self, countour_points):
        self.points = countour_points
        self.center, self.tangent = blob_mean_and_tangent(countour_points)
        self.angle = np.arctan2(self.tangent[1], self.tangent[0])
        self.rect = cv2.minAreaRect(countour_points)
        self.drawbox = cv2.boxPoints(self.rect)
        self.drawbox = np.intp(self.drawbox)
        self.area = cv2.contourArea(countour_points)
        self.top = min(countour_points[:, 0, 1])
        self.bottom = max(countour_points[:, 0, 1])
        self.left = min(countour_points[:, 0, 0])
        self.right = max(countour_points[:, 0, 0])
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.color = None

    def draw_on(self, img):
        cv2.drawContours(img, [self.drawbox], 0, (0, 255, 0), 2)
        cv2.drawContours(img, [self.points], 0, (0, 0, 255), 2)
        cv2.line(
            img,
            tuple(np.int32(self.center)),
            tuple(np.int32(self.center + 50 * self.tangent)),
            (255, 0, 0),
            5,
        )


def init_zones(staff_pred: np.ndarray, splits: int) -> Tuple[np.ndarray, int, int, int]:
    ys, xs = np.where(staff_pred > 0)

    # Define left and right bound
    accum_x = np.sum(staff_pred, axis=0)
    accum_x = accum_x / np.mean(accum_x)
    half = round(len(accum_x) / 2)
    right_bound = min(max(xs) + 50, staff_pred.shape[1])
    left_bound = max(min(xs) - 50, 0)
    for i in range(half + 10, len(accum_x)):
        if np.mean(accum_x[i - 10 : i]) < 0.1:
            right_bound = i
            break
    for i in range(half - 10, 0, -1):
        if np.mean(accum_x[i : i + 10]) < 0.1:
            left_bound = i
            break

    bottom_bound = min(max(ys) + 100, len(staff_pred))
    step_size = round((right_bound - left_bound) / splits)
    zones = []
    for start in range(left_bound, right_bound, step_size):
        end = start + step_size
        if right_bound - end < step_size:
            end = right_bound
            zones.append(range(start, end))
            break
        zones.append(range(start, end))
    return zones, left_bound, right_bound, bottom_bound


def find_contours(binary, original):
    cv2contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [Contour(c) for c in cv2contours]
    mostly_horizontal = [c for c in contours if abs(c.angle) < np.pi / 16]
    result = []

    #final = np.zeros(original.shape,np.uint8)
    mask = np.zeros(binary.shape,np.uint8)

    for contour in mostly_horizontal:
        if contour.area < 5000:
            continue
        result.append(contour)
        mask[...]=0
        cv2.drawContours(mask,[contour.points],-1,255,-1)
        contour.color = cv2.mean(original,mask)
        #cv2.drawContours(final,[contour.points],-1,cv2.mean(original,mask),-1)

    colors = [c.color for c in result]
    average_color, stdtdev_color = calculate_mean_and_stddev_for_colors(colors)
    result = filter_countours_by_color(result, average_color, stdtdev_color)
    # Perhaps better: Take relative y to closest neighbor, x start, area (normalized to median area) and color (median)

    return result

def filter_countours_by_color(contours, average_color, stddev_color):
    min_stddev = (30, 30, 30, 0)
    limit = np.maximum(min_stddev, stddev_color)
    return [c for c in contours if c.color is not None and np.all(np.abs(c.color - average_color) <= 1.2 * limit)]

def calculate_mean_and_stddev_for_colors(colors):
    return np.mean(colors, axis=0), np.std(colors, axis=0)

def extend_zone(zone1: range, zone2: range):
    if zone1 is None:
        return zone2
    if zone2 is None:
        return zone1
    return range(min(zone1.start, zone2.start), max(zone1.stop, zone2.stop))


def split_contours_in_zones(contours: list[Contour], zones):
    result = []
    for contour in contours:
        unaccounted_zone = None
        for zone in zones:
            full_range = extend_zone(unaccounted_zone, zone)
            points_in_zone = [p for p in contour.points if p[0][0] in full_range]
            if len(points_in_zone) == 0:
                continue
            contour_in_zone = Contour(np.array(points_in_zone))
            if contour_in_zone.area > 1000:
                result.append(contour_in_zone)
                unaccounted_zone = None
            else:
                unaccounted_zone = full_range
    return result


def draw_contours(contours, img):
    result = img.copy()
    for contour in contours:
        contour.draw_on(result)
    return result


class Rating:
    def __init__(self, contours: list[Contour], contours_by_zone: list[Contour]):
        self.staff_lines = len(contours)
        self.angles = sum([c.angle for c in contours_by_zone]) / len(contours_by_zone)


def detect_staffs(img, debug=True):
    #img = cv2.imread(filename)
    binary = image_threshold(img)
    if debug:
        cv2.imwrite("sheetmusic_binary.jpeg", binary)
    shapes = get_large_shapes(binary)
    if debug:
        cv2.imwrite("sheetmusic_shapes.jpeg", shapes)
    contours = find_contours(shapes, img)
    zones, _, _, _ = init_zones(binary, 8)
    contours_by_zone = split_contours_in_zones(contours, zones)
    contours_by_zone.extend(split_contours_in_zones(contours, list(reversed(zones))))
    if debug:
        contour_img = draw_contours(contours_by_zone, img)
        cv2.imwrite("sheetmusic_contours.jpeg", contour_img)
        cv2.imwrite("sheetmusic.jpeg", binary)
    return contours_by_zone, Rating(contours, contours_by_zone)

def crop_staff_lines(img, debug=True):
    binary = image_threshold(img)
    shapes = get_large_shapes(binary)
    contours = find_contours(shapes, img)
    if debug:
        contour_img = draw_contours(contours, img)
        cv2.imwrite("sheetmusic_crop.jpeg", binary)
        cv2.imwrite("sheetmusic_crop_contours.jpeg", contour_img)
    margin = 50
    left = max(min([c.left for c in contours]) - margin, 0)
    right = min(max([c.right for c in contours]) + margin, img.shape[1])
    top = max(min([c.top for c in contours]) - margin, 0)
    bottom = min(max([c.bottom for c in contours]) + margin, img.shape[0])
    return img[top:bottom, left:right]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python staffline_detection.py <image>")
        sys.exit(1)

    contours, result = detect_staffs(sys.argv[1], True)
    print(result.staff_lines, result.angles)
