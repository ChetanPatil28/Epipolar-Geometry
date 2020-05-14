import os

import numpy as np
from EstimateFundamentalMatrix import *
from FindEpipolarLines import *
import cv2
from FastHomography import warpHomo

def compute_epipole(F):
    print("F is ",F.shape)
    e1 = np.linalg.svd(F.T)[-1]
    e1 = e1[-1, :]/e1[-1, :][-1]

    e2 = np.linalg.svd(F)[-1]
    e2 = e2[-1, :]/e2[-1, :][-1]

    print("E shape is ",e1.shape,e2.shape)
    return e1,e2


def rectify_points(H, points):
    points = points.T
    for i in range(points.shape[1]):
        points[:, i] = np.dot(H, points[:, i])
        # convert the points to cartesian
        points[:, i] = points[:, i] / points[:, i][-1]

    return points

def find_H1_H2(e2, F, img2, pts1, pts2):

    w,h = img2.shape[:2]

    T = np.identity(3)
    T[0][2] = -1.0 * w / 2
    T[1][2] = -1.0 * h / 2

    e = T.dot(e2)
    print("e is ",e.shape)
    e1_prime = e[0]
    e2_prime = e[1]
    if e1_prime >= 0:
        alpha = 1.0
    else:
        alpha = -1.0

    R = np.identity(3)

    norm = np.sqrt(e1_prime**2 + e2_prime**2)
    R[0][0] = alpha * e1_prime / norm
    R[0][1] = alpha * e2_prime /norm
    R[1][0] = - alpha * e2_prime / norm
    R[1][1] = alpha * e1_prime / norm

    f = R.dot(e)[0]
    G = np.identity(3)
    G[2][0] = - 1.0 / f

    H2 = np.linalg.inv(T).dot(G.dot(R.dot(T)))

    e_prime = np.asarray([ [0, -e2[2], e2[1] ],
                           [e2[2], 0 , -e2[0]],
                           [-e2[1], e2[0], 0 ] ])

    v = np.array([1, 1, 1])
    M = e_prime.dot(F) + np.outer(e2, v)

    points1_hat = H2.dot(M.dot(pts1.T)).T
    points2_hat = H2.dot(pts2.T).T

    W = points1_hat / points1_hat[:, 2].reshape(-1, 1)
    b = (points2_hat / points2_hat[:, 2].reshape(-1, 1))[:, 0]
    # Minimisizng the least-squares to get the Matching-Transform..!
    a1, a2, a3 = np.linalg.lstsq(W, b)[0]
    HA = np.identity(3)
    HA[0] = np.array([a1, a2, a3])

    H1 = HA.dot(H2).dot(M)
    return H1, H2


def image_rectification(im1, im2, points1, points2):

    # Compute fundamental matrix
    F = find_epilines(points1, points2)
    # Compute epipoles
    e1, e2 = compute_epipole(F)

    print("Epipole shapes are ",e1.shape,e2.shape)

    points1 = np.column_stack((points1,np.ones(len(points1))))
    points2 = np.column_stack((points2,np.ones(len(points2))))
    # Compute homography
    H1, H2 = find_H1_H2(e2, F.T, im2, points1, points2)
    # rectify images
    rectified_im1 = warpHomo(np.linalg.inv(H1), im1)
    rectified_im2 = warpHomo(np.linalg.inv(H2), im2)
    # rectify points
    new_cor1 = rectify_points(H1, points1)
    new_cor2 = rectify_points(H2, points2)

    return rectified_im1, rectified_im2, new_cor1, new_cor2, F


def plot_epilines(pts_2dA, pts_2dB, img_a, img_b, F):
    eplines_a = get_lines(pts_2dB, F, from_where=2)
    eplines_b = get_lines(pts_2dA, F, from_where=1)

    n, m, _ = img_a.shape
    leftmost = np.cross([0, 0, 1], [n, 0, 1])
    rightmost = np.cross([0, m, 1], [n, m, 1])
    for i in range(len(eplines_a)):
        line_a, line_b = eplines_a[i], eplines_b[i]
        pt_a, pt_b = pts_2dA[i], pts_2dB[i]

        color = tuple(np.random.randint(0, 255, 3).tolist())
        leftmost_a = np.cross(line_a, leftmost)
        rightmost_a = np.cross(line_a, rightmost)
        leftmost_a = (leftmost_a[:2] / leftmost_a[2]).astype(int)
        rightmost_a = (rightmost_a[:2] / rightmost_a[2]).astype(int)
        cv2.line(img_a, tuple(leftmost_a[:2]), tuple(rightmost_a[:2]), color, thickness=1)
        cv2.circle(img_a, tuple(map(int, pt_a)), 4, color, -1)

        leftmost_b = np.cross(line_b, leftmost)
        rightmost_b = np.cross(line_b, rightmost)
        leftmost_b = (leftmost_b[:2] / leftmost_b[2]).astype(int)
        rightmost_b = (rightmost_b[:2] / rightmost_b[2]).astype(int)
        cv2.line(img_b, tuple(leftmost_b[:2]), tuple(rightmost_b[:2]), color, thickness=1)
        cv2.circle(img_b, tuple(map(int, pt_b)), 4, color, -1)

    return img_a, img_b

if __name__ == '__main__':
    TwoD_fileA = os.path.join(os.getcwd(), "Input/2d_pts_a.txt")
    TwoD_fileB = os.path.join(os.getcwd(), "Input/2d_pts_b.txt")

    # TwoD_fileA = os.path.join(os.getcwd(), "Input/cor1.npy")
    # TwoD_fileB = os.path.join(os.getcwd(), "Input/cor2.npy")

    img_a = cv2.imread('Input/pic_a.jpg', cv2.IMREAD_COLOR)
    img_b = cv2.imread('Input/pic_b.jpg', cv2.IMREAD_COLOR)

    # img_a = cv2.imread('Input/dino0.png', cv2.IMREAD_COLOR)
    # img_b = cv2.imread('Input/dino1.png', cv2.IMREAD_COLOR)
    try:
        pts_2dA = load_points(TwoD_fileA)
        pts_2dB = load_points(TwoD_fileB)
    except UnicodeError:
        pts_2dA = np.load(TwoD_fileA).T[:,:2]
        pts_2dB = np.load(TwoD_fileB).T[:,:2]


    # F = find_epipoles(pts_2dA, pts_2dB, img_a, img_b,show_lines = True)
    # print("Estimated Fundamental-matrix is..\n ",F)


    rect_img1, rect_img2, epilines1, epilines2, F = image_rectification(img_a, img_b, pts_2dA, pts_2dB)

    # rect_img1, rect_img2 = plot_epilines(epilines1, epilines2, rect_img1, rect_img2, F)



    cv2.imwrite("Result/Rectified_left.png",rect_img1)
    cv2.imwrite("Result/Rectified_right.png", rect_img2)

    # cv2.imshow("rect1,",rect_img1)
    # cv2.imshow("rect2,",rect_img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()