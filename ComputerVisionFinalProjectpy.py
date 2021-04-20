## Computer Vision Final Project
# May 2020
# Gazi13
##

import cv2 as cv
import numpy as np
import time
from scipy import signal
import random as rng



def optical_flow(I1g, I2g, window_size, tau=1e-2):
 
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0],1)) # get b here
            A = np.vstack((Ix, Iy)).T # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b) 
                u[i,j]=nu[0]
                v[i,j]=nu[1]
 
    return (u,v)


h,w = 320,240
fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter("hahaha-1.mp4", fourcc, 25, (h,w), True)
    
    
cap = cv.VideoCapture("b.mp4")

ret, first_frame = cap.read()
first_frame = cv.resize(first_frame, (h,w))

prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)



mask = np.zeros_like(first_frame)
mask_contour = np.zeros_like(first_frame)
kernel = np.ones((5, 5), np.uint8)

tracker = cv.MultiTracker_create()
init_once = False

checkPoint = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        break
    frame = cv.resize(frame, (h,w))
    img_org = frame#.copy()
    checkPoint+=1


    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    
    if checkPoint%50 == 0 or checkPoint==1:
        prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        mask = np.zeros_like(first_frame)
        mask_contour = np.zeros_like(gray)
    
        init_once= False
        tracker = cv.MultiTracker_create()
        continue
        #prev = prev[1:100]


    if not init_once:

        u,v = optical_flow(prev_gray, gray, 15, tau=1e-2)
        #print(u.shape,"--",next.shape)#(480, 640) -- (106, 1, 2)
        if u is None:
            continue
        for i,(uu,vv) in enumerate(zip(u,v)):
            for j,(u,v) in enumerate(zip(uu,vv)):
                u = int(u)
                v = int(v)
                if u!=0 or v !=0:
                    mask_contour = cv.circle(mask_contour, (j, i), 2, 255, -1)

        #mask_contour = cv.cvtColor(mask_contour, cv.COLOR_BGR2GRAY)
        erode = cv.erode(mask_contour, kernel,iterations=2)
        dilate = cv.dilate(erode, kernel,iterations=4)
        cv.imshow("dilate", dilate)
        contours, hierarchy = cv.findContours(dilate,cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
        print(len(contours))

        
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        for i, c in enumerate(contours):
            if not hierarchy[0,i,3] < 1 and hierarchy[0,i,0] == -1:
                continue
            contours_poly[i] = cv.approxPolyDP(c, 3, True)
            boundRect[i] = cv.boundingRect(contours_poly[i])
        for i in range(0,len(contours)):
            if boundRect[i] == None \
               or boundRect[i][2]>(600) \
               or boundRect[i][3]>(600) \
               or boundRect[i][2]<15 \
               or boundRect[i][3]<15 \
               or min(boundRect[i][2],boundRect[i][3])/max(boundRect[i][2],boundRect[i][3])<0.2: # for too high or wide
                continue
            
             
            #cv.rectangle(img_org, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)
            ok = tracker.add(cv.TrackerMIL_create(),frame, tuple(boundRect[i]))
        init_once = True
        
    ok,objects = tracker.update(frame)
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    for newbox in objects:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        center = (int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2))

        if p2[1]>(frame.shape[0]-10)\
           or p1[1]<10:
            continue
        cv.rectangle(img_org, p1, p2, (0,0,255),3)
        mask = cv.circle(mask, center, 2, (0,255,0), -1)
    if len(objects)<1:
        init_once=False


    # Overlays the optical flow tracks on the original frame
    output = cv.add(img_org, mask)
    # Updates previous frame
    prev_gray = gray.copy()

    cv.imshow("sparse optical flow", output)
    #out.write(output)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv.destroyAllWindows()
