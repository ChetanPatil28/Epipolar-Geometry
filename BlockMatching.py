### ITER-5

### shifting the left-array by two positons so verify i the minima occur diagonally.!

import cv2
import numpy as np
import time
from tqdm import tqdm


def compute_norm(arr):
    nume = arr-arr.mean()
    deno  = np.sqrt((nume**2).sum())
    return nume/deno

max_ssd = 10
max_d = 15
# width = 10
hb = 4
# height = 3
# comps = width -2*hb + 2
# left=  np.random.randint(0,34,size = (height,width))
# right = np.roll(left, -2,axis = 1)

left = cv2.imread("Inputs/tsukuba_l.png",0)
right = cv2.imread("Inputs/tsukuba_r.png",0)

subsampling = False

if subsampling:
    left = cv2.pyrDown(left)
    right = cv2.pyrDown(right)


height, width = left.shape

disparity_img = np.zeros(shape = (height,width))

tik = time.time()

for h in tqdm(range(hb,height-hb)):
    lshift = 0
    DSI = np.ones(shape = (width-2*hb,max_d))
    for w in range(hb,width-hb):
        if (w-hb)>=max_d: lshift+=1
        for d in range(0,max_d):
            rshift = w-d-hb
            if(rshift>=0):
                left_patch = left[h-hb:h+hb+1,w-hb:w+hb+1].copy()
                right_patch = right[h-hb:h+hb+1,w-hb-rshift+lshift:w+hb+1-rshift+lshift].copy()
                ssd = ((compute_norm(left_patch)-compute_norm(right_patch))**2).sum()+0.0001
                index  = rshift-lshift
            else:
                index = d
                ssd = max_ssd
                pass
            DSI[w-hb,index] = ssd
        disparity_img[h,hb:-hb] = np.argmin(DSI, axis = 1)


print("Time took ",time.time()-tik)

final_img = ((disparity_img/disparity_img.max())*255).astype(np.uint8)


if subsampling:
    final_img = cv2.pyrUp(final_img)

cv2.imwrite("Outputs/Estimated_Disparity.png",final_img)

cv2.imshow("Estimated_Disparity",final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
