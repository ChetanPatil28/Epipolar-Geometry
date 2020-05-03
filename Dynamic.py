### LAST ITERATION
### ITER-5

### comments are written for better understanding.....!

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
cost  = 100
# height = 3
# comps = width -2*hb + 2
# left=  np.random.randint(0,34,size = (height,width))
# right = np.roll(left, -2,axis = 1)

print("Block size is ",2*hb+1)
print("Max disparity is ",max_d)

left = cv2.imread("Inputs/tsukuba_l.png",0)
right = cv2.imread("Inputs/tsukuba_r.png",0)

height, width = left.shape

disparity_img = np.zeros(shape = (height,width))

# print("Left is \n",left)
# print("Right is \n",right)

tik = time.time()

# iterate thru height.!
for h in tqdm(range(hb,height-hb)):
#     print(h)
    lshift = 0
    DSI = np.zeros(shape = (width-2*hb,max_d))
    # print("DSI shape is ",DSI.shape, h)
    #for every height, u need a DSI so u init it 
    for w in range(hb,width-hb):
        if (w-hb)>=max_d: lshift+=1
        # for every pixel in left-img, scan upto dmax and compute ssd, fill this ssd inside DSI
        for d in range(0,max_d):
            rshift = w-d-hb
            if(rshift>=0):
                left_patch = left[h-hb:h+hb+1,w-hb:w+hb+1].copy()
                right_patch = right[h-hb:h+hb+1,w-hb-rshift+lshift:w+hb+1-rshift+lshift].copy()
#                 print("Left-patch is \n{}\nRigth patch is \n{}".format(left_patch,right_patch))
                ssd = ((compute_norm(left_patch)-compute_norm(right_patch))**2).sum()
#                 print("SSD is {}\n".format(ssd))

                index  = rshift-lshift
            else:
                index = d
                ssd = max_ssd
                pass
            DSI[w-hb,index] = ssd
    
    DSI[DSI==0] = np.inf
    
    for i in range(1,DSI.shape[0]):
        for j in range(i,min(DSI.shape[1],i+max_d)):
            DSI[i,j-i] = DSI[i,j-i] + min(DSI[i-1,:] + cost*np.abs(np.arange(0,DSI.shape[1])-(i-j)))


#     curr_id = np.argmin(DSI[-1])
#     curr_cost = DSI[-1,curr_id]
# #     print(curr_cost)
#     l = [curr_id]
# #     print(l)
#     for row in range(DSI.shape[0]-2,-1,-1):
#     #     print(M[row],np.abs(curr_cost-M[row]))
#         curr_id = np.argmin(np.abs(curr_cost-DSI[row]))
#         curr_cost = DSI[row,curr_id]
#         l.append(curr_id)        
    
    # c = np.zeros(DSI.shape[0])
    # for i in range(DSI.shape[0]-1,-1,-1):
    #     b = np.argmin(DSI[i,:])
    #     c[i] = i

    disparity_img[h,hb:-hb] = np.argmin(DSI, axis = 1)
    # disparity_img[h,hb:-hb] = c

#         print("One pixel disparity compared at\n")
# print(np.round(DSI,decimals = 2).T)

print("Time took ",time.time()-tik)

final_img = ((disparity_img/disparity_img.max())*255).astype(np.uint8)

cv2.imwrite("Outputs/Disparity_dynamic.png",final_img)

cv2.imshow("final",final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
