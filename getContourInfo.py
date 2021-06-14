import cv2
import numpy as np
import copy

kernel_size = 9
lowThreshold = 200
highThreshold = 300
CANNY_LOW_THRESHOLD_RATIO = 0.66
CANNY_RATIO_CONTROL_THRESHOLD = 0.1 / CANNY_LOW_THRESHOLD_RATIO
ratio = 2

def getContourInfo(gray_edges):
    height,width = gray_edges.shape
    long = height if height>width else width
    short = width if long == height else height
    (result, hierarchy) = cv2.findContours(gray_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #retr_external:只檢測外輪廓, retr_tree:建立等級樹
    print("number of contour result:",np.size(result))

    new = []
    hier = []
    size = []
    contourInfo = []
    #紀錄hierarchy和座標長寬
    #temp = []
    for a,b,c in zip(result, hierarchy[0].tolist(), range(len(result))):
        
        (x, y, w, h) = cv2.boundingRect(a)
        
        if b[3] == -1:
            new.append(a)
            hier.append([c,b])
            #temp.append(c)
            #記錄第一層輪廓的編號
        elif b[2] != -1:
            new.append(a)
            hier.append([c,b])
            #加入有父輪廓且有子輪廓的
        elif w >= short/15 and h>=short/15 and np.size(result)<250:
            new.append(a)
            hier.append([c,b])
            #加入最底層但是rectangle夠大的
            #只在畫面沒那麼多contour時加入
    for c in new:
        (x, y, w, h) = cv2.boundingRect(c)
        size.append([x,y,w,h])
    
    print("number of contours after filted by hierarchy:",len(hier))
    contourInfo = [list(item) for item in zip(hier,size)]
    #合併hier和size [[輪廓編號, 輪廓hier資訊],[輪廓座標]]
    for c in contourInfo.copy():
        x,y,w,h = c[1]
        if (w/h<=0.15 and h<=height/4) or w/h<=0.05 or (h/w<=0.15 and w<=width/4) or h/w<=0.05:
            contourInfo.remove(c)
            #print('remove:', c)

            #去掉直長條或橫長條
    print('number of contours after filted by size:', len(contourInfo))

    return result, hierarchy, contourInfo

##delete noises in big image
def removeImgNoise(contourInfo, img_gray):
    temp = contourInfo.copy()
    pixelInfo = []
    h,w = img_gray.shape
    dist_factor = w*0.01 if w<h else h*0.01
    for c in temp:
        x0,y0,w0,h0 = c[1]
        img_crop = img_gray[y0:y0+h0, x0:x0+w0].copy()
        unique, counts = np.unique(img_crop, return_counts=True)
        pixelInfo.append([max(counts), img_crop.size])
        
    for c,e in zip(temp, pixelInfo):
        xc,yc,wc,hc = c[1]
        maxc, sizec = e
        if (maxc/sizec<=0.43 and hc<=0.95*h and wc<=0.95*w) or maxc/sizec<=0.26:
            for d, f in zip(temp, pixelInfo):

                xd,yd,wd,hd = d[1]
                maxd, sized = f
                if hd<hc and wd<wc and xd>=xc and xd+wd<=xc+wc and yd>=yc and yd+hd<=yc+hc:
                    if maxd/sized<=0.43 and wc-wd>=3*dist_factor and hc-hd>=3*dist_factor:
                        
                        if d in contourInfo: 
                            contourInfo.remove(d)
                            #print('remove:',d,f)
    # for c,e in zip(temp, pixelInfo):
    #     print(c,e)
                        
    return contourInfo


def addDilate(imgData, size):
    dilationSize = size 
    kernel = np.ones((2 * dilationSize + 1, 2 * dilationSize + 1), np.uint8)
    img_dilation = cv2.dilate(imgData, kernel, iterations=1)
    return img_dilation

def updateLowHighThreshold(imgData):
    imgHist = cv2.equalizeHist(imgData)
    mean = np.average(imgHist)
    lowThreshold = int(CANNY_LOW_THRESHOLD_RATIO * mean * CANNY_RATIO_CONTROL_THRESHOLD)
    highThreshold = int(CANNY_LOW_THRESHOLD_RATIO * ratio * mean * CANNY_RATIO_CONTROL_THRESHOLD)
    
    return (lowThreshold, highThreshold, mean)

def splitCanny(img, kernel_size):
    blue = cv2.split(img)[0]
    green = cv2.split(img)[1]
    red = cv2.split(img)[2]
    
    blur_blue = cv2.GaussianBlur(blue,(kernel_size, kernel_size), 0)
    blur_green = cv2.GaussianBlur(green,(kernel_size, kernel_size), 0)
    blur_red = cv2.GaussianBlur(red,(kernel_size, kernel_size), 0)
    
    #bgr
    blue_edges = cv2.Canny(blur_blue, updateLowHighThreshold(blue)[0], updateLowHighThreshold(blue)[1])
    green_edges = cv2.Canny(blur_green, updateLowHighThreshold(green)[0], updateLowHighThreshold(green)[1])
    red_edges = cv2.Canny(blur_red, updateLowHighThreshold(red)[0], updateLowHighThreshold(red)[1])
    edges = blue_edges | green_edges | red_edges
    
    edges = addDilate(edges, 8)
    #dilate or not
    
    return edges

def remove_small(result, low_threshold):
    #去掉contour點數少的
    result.sort(reverse=True, key=len)
    new = []
    for i in result:
        new.append(i)
        if np.size(i) <= low_threshold:
            break
    return new
