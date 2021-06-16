#from  matplotlib import pyplot as plt
from imp import reload 
import sys
import cv2
import numpy as np
from  matplotlib import pyplot as plt

from getContourInfo import *
reload(sys.modules['getContourInfo'])
from getContourInfo import *
from addText import *
reload(sys.modules['addText'])
from addText import *

from generateDict import *
reload(sys.modules['generateDict'])
from generateDict import *

import tesserocr
import pytesseract

from PIL import Image
import concurrent.futures
import queue

import time
import os

import math 
from combineContour import *
reload(sys.modules['combineContour'])
from combineContour import *

from PIL import Image, ImageDraw, ImageFont

def allProcess(IMAGE_NUMBER, experiment):
    if not(experiment):
        img = cv2.imread('iosScreenshot/IMG_'+IMAGE_NUMBER+'.png', 1)
    else:
        img = cv2.imread('experiment/'+IMAGE_NUMBER+'.png', 1)
    height,width,_ = img.shape

    print('image size: ', height, '*', width)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lowThreshold, highThreshold, mean = updateLowHighThreshold(gray)
    print("lowThreshold, highThreshold, mean:",lowThreshold, highThreshold, mean)

    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    gray_edges = cv2.Canny(blur_gray, lowThreshold, highThreshold)

    
    mean_canny = np.average(gray_edges)
    print('canny mean:',mean_canny)

    if mean_canny<=7:
        i=7
    elif 7<mean_canny<17:
        i=5
    else:
        i=3

    gray_edges = addDilate(gray_edges, i)
    #加深edge
    mean_canny = np.average(gray_edges)

    print('canny mean after dilate by '+str(i)+' :',mean_canny)

    cv2.imwrite('result/canny.jpg', gray_edges)

    result, hierarchy, contourInfo = getContourInfo(gray_edges)
    contourInfo = removeImgNoise(contourInfo, gray)
    print('number of contours after filted out img noises:', len(contourInfo))

    clone = img.copy()
    for c in result:
        (x, y, w, h) = cv2.boundingRect(c)
        #if w>=30 and h>=30: 
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #長方形框出contour

    if not(experiment): cv2.imwrite('result/'+IMAGE_NUMBER+'.png', clone)

    #利用hierarchy 資訊篩選重要的contours
    #print(result)
    #print(len(hierarchy[0]))
    #print(hierarchy[0])
    #同層下一個輪廓的序號、同層上一個輪廓的序號、子輪廓的序號、父輪廓的序號。
    #contourInfo: [[輪廓編號, 輪廓hier資訊],[輪廓座標]]
    clone2 = img.copy()
    for c in contourInfo:
        x,y,w,h = c[1]
        cv2.rectangle(clone2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(clone2, str(c[0][0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if not(experiment): 
        cv2.imwrite('result/better_'+IMAGE_NUMBER+'.png', clone2)
    else:
        cv2.imwrite('experiment/better_'+IMAGE_NUMBER+'.png', clone2)


    NUM_THREADS = os.cpu_count()
    NUM_THREADS = 8
    print('Logical Processors: ',NUM_THREADS)
    #tesserocr_queue = queue.Queue()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.fastNlMeansDenoising(img_gray)
    #denoising


    for t_num in range(NUM_THREADS):
        tesserocr_queue.put([t_num, tesserocr.PyTessBaseAPI(lang='eng+chi_tra',psm=6)])

    start = time.time()

    #from itertools import repeat
    with concurrent.futures.ThreadPoolExecutor(NUM_THREADS) as executor:
        futures = {executor.submit(addText, c, img_gray): c for c in contourInfo}
        concurrent.futures.wait(futures)
    #     futures = executor.map(addText, contourInfo, repeat(img_gray))
    #     concurrent.futures.wait(futures)
        
        
    end = time.time()
    print('took tot: ' + str(end - start))
    for _ in range(NUM_THREADS):
        t_num, api = tesserocr_queue.get(block=True)
        api.End()
        
        



    sizeOG = img.shape
    contourInfo_new = contourInfo.copy()
    contourInfo_new.reverse()

    contourInfo_new = combineContour(contourInfo_new, sizeOG)

    print('before combine: ', len(contourInfo))
    print('after combine1: ', len(contourInfo_new))
    contourInfo_new = combineContour(contourInfo_new, sizeOG)
    print('after combine2: ', len(contourInfo_new))
    contourInfo_new = combineContour(contourInfo_new, sizeOG)
    print('after combine3: ', len(contourInfo_new))



    clone4 = img.copy()
    for i,c in enumerate(contourInfo_new):
        x,y,w,h = c[1]
        cv2.rectangle(clone4, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(clone4, str(c[0][0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    if not(experiment): cv2.imwrite('result/combined_'+IMAGE_NUMBER+'.jpg', clone4)
        


    clone5 = img.copy()
    clone5[:][:][:] = 255
    for i, c in enumerate(contourInfo_new):
        x,y,w,h = c[1]
        cv2.rectangle(clone5, (x, y), (x + w, y + h), (0, 255, 0), 2)

    clone5 = cv2.cvtColor(clone5, cv2.COLOR_BGR2RGB)
    pil5 = Image.fromarray(clone5)
    for i, c in enumerate(contourInfo_new):
        x,y,w,h = c[1]
        if c[2][1] >=70:
            draw = ImageDraw.Draw(pil5)
            fontText = ImageFont.truetype('msyhbd.ttf' , 20, encoding="utf-8")
            draw.text((x, y), c[2][0], (0, 0, 0), font=fontText)


    if not(experiment): pil5.save('result/contour&text_'+IMAGE_NUMBER+'.jpg')

    clone6 = img.copy()
    mainDict, temp = generateJsonDict('test.json',img, contourInfo_new, sizeOG, clone6)
    print('temp length: ', len(temp))
    if not(experiment): cv2.imwrite('result/secondLayer_'+IMAGE_NUMBER+'.jpg', clone6)

    return contourInfo_new


