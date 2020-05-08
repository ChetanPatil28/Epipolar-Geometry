import numpy as np
import cv2
from EstimateFundamentalMatrix import *


def find_epipoles(pts_2dA, pts_2dB,img_a,img_b):

    ### Normalizing the points, because normalizing is good. (as stated in the 8-point algorithm)
    mean_A = np.mean(pts_2dA,axis = 0)
    scale_A = np.linalg.norm((pts_2dA-mean_A))/len(pts_2dA)

    mean_B = np.mean(pts_2dB, axis=0)
    scale_B = np.linalg.norm((pts_2dB - mean_B)) / (2*len(pts_2dB))


    scale_A = np.sqrt(2)/scale_A
    scale_B = np.sqrt(2)/scale_B


    norm_pts_A = (pts_2dA - mean_A)* scale_A;
    norm_pts_B = (pts_2dB - mean_B) * scale_B;

    denorm_A = np.asarray([[scale_A, 0, -scale_A*mean_A[0]],
                           [0, scale_A, -scale_A*mean_A[1]],
                           [0, 0, 1]], dtype = np.float32)

    denorm_B = np.asarray([[scale_B, 0, -scale_B*mean_B[0]],
                           [0, scale_B, -scale_B*mean_B[1]],
                           [0, 0, 1]], dtype = np.float32)


    print("denorm_A\n",denorm_A)

    print("denorm_B\n",denorm_B)


    norm_pts_A = np.column_stack((norm_pts_A, np.ones(norm_pts_A.shape[0])))
    norm_pts_B = np.column_stack((norm_pts_B, np.ones(norm_pts_B.shape[0])))
    print("NORM-pts shape are ",norm_pts_A.shape,norm_pts_B.shape)

    
    ## Our fundamental-matrix is rank-2, so we need to set the smallest singular-value to 0.
    Fundam = svd_decompose(norm_pts_A, norm_pts_B, rank = 2)

    ## Denormalizing the F-matrix with our previously stored stats.
    F = np.dot(denorm_B.T, np.dot(Fundam, denorm_A))

    print("Estimated Fundamental-Matrix is \n", F)



    cv_F,_ = fundamental, inliers = cv2.findFundamentalMat(pts_2dA,pts_2dB)
    print("CV2 Fundam is \n",cv_F)

    pts_a = np.column_stack((pts_2dA, np.ones(norm_pts_A.shape[0])))
    pts_b = np.column_stack((pts_2dB, np.ones(norm_pts_B.shape[0])))

    print("Diff is Fund estimation",np.linalg.norm(F-cv_F))

    eplines_a = np.dot(F.T, pts_a.T).T
    eplines_b = np.dot(F, pts_b.T).T

    cv_epilines_a = np.dot(cv_F.T, pts_a.T).T
    cv_epilines_b = np.dot(cv_F, pts_b.T).T

    print("diff-a ", np.linalg.norm(eplines_a-cv_epilines_a))
    print("diff-b ",np.linalg.norm(eplines_b-cv_epilines_b))

    n, m, _ = img_a.shape
    line_L = np.cross([0, 0, 1], [n, 0, 1])
    line_R = np.cross([0, m, 1], [n, m, 1])
    for line_a, line_b in zip(eplines_a, eplines_b):
        P_a_L = np.cross(line_a, line_L)
        P_a_R = np.cross(line_a, line_R)
        P_a_L = (P_a_L[:2] / P_a_R[2]).astype(int)
        P_a_R = (P_a_R[:2] / P_a_R[2]).astype(int)
        cv2.line(img_a, tuple(P_a_L[:2]), tuple(P_a_R[:2]), (255, 0, 0), thickness=1)
        P_b_L = np.cross(line_b, line_L)
        P_b_R = np.cross(line_b, line_R)
        P_b_L = (P_b_L[:2] / P_b_R[2]).astype(int)
        P_b_R = (P_b_R[:2] / P_b_R[2]).astype(int)
        cv2.line(img_b, tuple(P_b_L[:2]), tuple(P_b_R[:2]), (255, 0, 0), thickness=1)

    for line_a, line_b in zip(cv_epilines_a, cv_epilines_b):
        P_a_L = np.cross(line_a, line_L)
        P_a_R = np.cross(line_a, line_R)
        P_a_L = (P_a_L[:2] / P_a_R[2]).astype(int)
        P_a_R = (P_a_R[:2] / P_a_R[2]).astype(int)
        cv2.line(img_a, tuple(P_a_L[:2]), tuple(P_a_R[:2]), (0, 255, 0), thickness=1)
        P_b_L = np.cross(line_b, line_L)
        P_b_R = np.cross(line_b, line_R)
        P_b_L = (P_b_L[:2] / P_b_R[2]).astype(int)
        P_b_R = (P_b_R[:2] / P_b_R[2]).astype(int)
        cv2.line(img_b, tuple(P_b_L[:2]), tuple(P_b_R[:2]), (0, 255, 0), thickness=1)



    cv2.imwrite('Epilines_A.png', img_a)
    cv2.imwrite('Epilines_B.png', img_b)


if __name__ == "__main__":
    TwoD_fileA = os.path.join(os.getcwd(), "Input/2d_pts_a.txt")
    TwoD_fileB = os.path.join(os.getcwd(), "Input/2d_pts_b.txt")


    img_a = cv2.imread('Input/pic_a.jpg', cv2.IMREAD_COLOR)
    img_b = cv2.imread('Input/pic_b.jpg', cv2.IMREAD_COLOR)
    pts_2dA = load_points(TwoD_fileA)
    pts_2dB = load_points(TwoD_fileB)


    find_epipoles(pts_2dA, pts_2dB, img_a, img_b)