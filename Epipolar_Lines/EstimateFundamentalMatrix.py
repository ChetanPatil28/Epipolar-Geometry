import numpy as np
import cv2
import os

def load_points(file):
    l = []
    with open(file,"r") as f:
        for pnt in f.readlines():
            l.append(pnt.strip().split())
        f.close()
    return np.asarray(l,dtype = np.float32)




def svd_decompose(pts_a,pts_b, rank =2):
    num_pts = pts_a.shape[0]
    xa = pts_a[:,0]
    ya = pts_a[:,1]
    xb = pts_b[:,0]
    yb = pts_b[:,1]
    ones = np.ones(num_pts)
    A = np.column_stack((xa*xb, ya*xb, xb, xa*yb, ya*yb, yb, xa, ya, ones))
    _,_,V = np.linalg.svd(A, full_matrices=True)
    F = V.T[:,-1]
    F = F.reshape((3,3))

    ## F is a rank-2 matrix actually, so we throw off the least eigen value by again decomposing the F
    # into rotation, squeeze, rotation, set last value of S to 0, then multiply them back, to get rank-2 matrix.
    if rank==2:
        U, S, V = np.linalg.svd(F)
        S[-1] = 0
        S = np.diag(S)
        F = np.dot(np.dot(U, S), V)
    return F
    # pass





if __name__=="__main__":
    pass

### This

# TwoD_fileA = os.path.join(os.getcwd(),"input/pts2d-pic_a.txt")
# TwoD_fileB = os.path.join(os.getcwd(),"input/pts2d-pic_b.txt")


# pts_2dA = load_points(TwoD_fileA)
# pts_2dB = load_points(TwoD_fileB)
# Fund = svd_decompose(pts_2dA,pts_2dB)

# print(Fund)
