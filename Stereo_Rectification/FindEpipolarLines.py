import numpy as np
import cv2
from EstimateFundamentalMatrix import *



def get_lines(points,F_matrix,from_where = 2):
    ## use from_where as 2, when u insert points in 2nd img.
    pts_homo = np.column_stack((points,np.ones(points.shape[0])))
    if from_where==2:
        lines = np.dot(pts_homo,F_matrix)
    else:
        lines = np.dot(pts_homo,F_matrix.T)
    
    lines/=np.sqrt(((lines[:,0])**2 + (lines[:,1])**2)).reshape(-1,1)
    return lines

def find_epipoles(pts_2dA, pts_2dB,img_a,img_b):

    ### Normalizing the points, because normalizing is good. (as stated in the 8-point algorithm)
    mean_A = np.mean(pts_2dA,axis = 0)
    scale_A = np.linalg.norm((pts_2dA-mean_A))/len(pts_2dA)

    mean_B = np.mean(pts_2dB, axis=0)
    scale_B = np.linalg.norm((pts_2dB - mean_B)) / (len(pts_2dB))


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

    norm_pts_A = np.column_stack((norm_pts_A, np.ones(norm_pts_A.shape[0])))
    norm_pts_B = np.column_stack((norm_pts_B, np.ones(norm_pts_B.shape[0])))
    
    ## Our fundamental-matrix is rank-2, so we need to set the smallest singular-value to 0.
    Fundam = svdecompose(norm_pts_A, norm_pts_B, rank = 2)

    ## Denormalizing the F-matrix with our previously stored stats.
    F = np.dot(denorm_B.T, np.dot(Fundam, denorm_A))

    print("Estimated Fundamental-Matrix is \n", F)

    eplines_a = get_lines(pts_2dB, F, from_where=2)
    eplines_b = get_lines(pts_2dA, F, from_where=1)
    

    n, m, _ = img_a.shape
    leftmost = np.cross([0, 0, 1], [n, 0, 1])
    rightmost = np.cross([0, m, 1], [n, m, 1])
    for i in range(len(eplines_a)):
        line_a, line_b = eplines_a[i], eplines_b[i]
        pt_a,pt_b = pts_2dA[i], pts_2dB[i]

        color = tuple(np.random.randint(0,255,3).tolist())
        leftmost_a = np.cross(line_a, leftmost)
        rightmost_a = np.cross(line_a, rightmost)
        leftmost_a = (leftmost_a[:2] / leftmost_a[2]).astype(int)
        rightmost_a = (rightmost_a[:2] / rightmost_a[2]).astype(int)
        cv2.line(img_a, tuple(leftmost_a[:2]), tuple(rightmost_a[:2]), color, thickness=1)
        cv2.circle(img_a, tuple(map(int,pt_a)), 4, color, -1)

        leftmost_b = np.cross(line_b, leftmost)
        rightmost_b = np.cross(line_b, rightmost)
        leftmost_b = (leftmost_b[:2] / leftmost_b[2]).astype(int)
        rightmost_b = (rightmost_b[:2] / rightmost_b[2]).astype(int)
        cv2.line(img_b, tuple(leftmost_b[:2]), tuple(rightmost_b[:2]), color, thickness=1)
        cv2.circle(img_b, tuple(map(int,pt_b)), 4, color, -1)



    cv2.imwrite('Result/Epilines_A.png', img_a)
    cv2.imwrite('Result/Epilines_B.png', img_b)


if __name__ == "__main__":
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


    find_epipoles(pts_2dA, pts_2dB, img_a, img_b)