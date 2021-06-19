import os, sys 
import platform 
import cv2 
import pytesseract
import tesserocr
from PIL import Image
import queue
import concurrent.futures
import numpy as np
import math 

if platform.system() == 'Windows': 
    pytesseract.pytesseract.tesseract_cmd = None 
    p = os.path.join("C:\\", "Program Files (x86)", "Tesseract-OCR", "tesseract.exe")
    if os.path.isfile(p): 
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

    if pytesseract.pytesseract.tesseract_cmd == None: 
        p = os.path.join("C:\\", "Program Files", "Tesseract-OCR", "tesseract.exe")
        if os.path.isfile(p): 
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    if pytesseract.pytesseract.tesseract_cmd == None: 
        p = os.path.join('C:\\', "Users", os.getlogin(), "AppData", "Local", "Tesseract-OCR", "tesseract.exe")
        if os.path.isfile(p): 
            pytesseract.pytesseract.tesseract_cmd \
            = r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getlogin())
    
    if pytesseract.pytesseract.tesseract_cmd == None: 
        print("Sorry that I could not find your tesseract.exe!")
        sys.exit(-1)

class screenImageAnalyzer: 
    def __init__(self): 
        
        self.langs = 'eng+chi_tra+chi_sim'
        # self.langs = 'eng+chi_tra'
        #self.ocr_config = '--psm 6' #for pytesseract
        self.ocr_config = 6 #for tesserocr
        self.countCVedContours = 0

        self.kernel_size = 9

        self.CANNY_LOW_THRESHOLD_RATIO = 0.66
        self.CANNY_RATIO_CONTROL_THRESHOLD = 0.1 / self.CANNY_LOW_THRESHOLD_RATIO
        self.ratio = 2
        self.NUM_THREADS = os.cpu_count()
        self.tesserocr_queue = queue.Queue()

        
        


    def getImageFromFile(self, inputImgPath): 
        self.cv2Input = cv2.imread(inputImgPath)
        return self.cv2Input 

    def getImage(self, cv2InputImg): 
        self.cv2Input = cv2InputImg 
        return self.cv2Input 

    def getVisionDict(self, x = None, y=None, width=None, height=None): 
        # get the actual image
        self.x = x
        self.y= y
        self.width = width
        self.height = height
        if x == None: 
            self.img = self.cv2Input[:]
            self.x = 0
            self.y = 0 
            self.width = self.cv2Input.shape[1]
            self.height = self.cv2Input.shape[0]
        else: 
            self.img = self.cv2Input[y:(y+height), x:(x+width)]

        print("********")

        self.sizeOG = self.cv2Input.shape
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        lowThreshold, highThreshold, mean = self.updateLowHighThreshold(gray)
        blur_gray = cv2.GaussianBlur(gray,(self.kernel_size, self.kernel_size), 0)
        gray_edges = cv2.Canny(blur_gray, lowThreshold, highThreshold)

        mean_canny = np.average(gray_edges)
        if mean_canny<=7:
            i=7
        elif 7<mean_canny<15:
            i=5
        else:
            i=3
        
        gray_edges = self.addDilate(gray_edges, i)
        contourInfo = self.getContourInfo(gray_edges, self.sizeOG)
        contourInfo = self.removeImgNoise(contourInfo, gray, self.sizeOG)
        

        

        img_gray = cv2.fastNlMeansDenoising(gray)

        #self.api = tesserocr.PyTessBaseAPI(lang='eng+chi_tra',psm=6)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = {executor.submit(self.addText, c, img_gray): c for c in contourInfo}
        #     concurrent.futures.wait(futures)
        #contourInfo = self.addTextInfo(contourInfo, img_gray)

        for _ in range(self.NUM_THREADS):
            self.tesserocr_queue.put(tesserocr.PyTessBaseAPI(lang=self.langs,psm=self.ocr_config))

        with concurrent.futures.ThreadPoolExecutor(self.NUM_THREADS) as executor:
            futures = {executor.submit(self.addText, c, img_gray): c for c in contourInfo}
            concurrent.futures.wait(futures)

        for _ in range(self.NUM_THREADS):
            api = self.tesserocr_queue.get(block=True)
            api.End()

        print("Add Text Info")

        contourInfo_new = contourInfo.copy()
        contourInfo_new.reverse()
        contourInfo_new = self.combineContour(contourInfo_new, self.sizeOG)
        contourInfo_new = self.combineContour(contourInfo_new, self.sizeOG)
        print('number of contours after combination: ', len(contourInfo_new))
        contourInfo_new = self.removeOverlap(contourInfo_new, self.sizeOG)
        contourInfo_new = self.removeOverlap(contourInfo_new, self.sizeOG)
        print('number of contours after removed overlapping: ', len(contourInfo_new))


        self.CVCanvas = self.img.copy()
        for i,c in enumerate(contourInfo_new):
            x,y,w,h = c[1]
            cv2.rectangle(self.CVCanvas , (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.conDict = self.generateJsonDict(self.img, contourInfo_new, self.sizeOG)
        print('Write to dict')

        print("The all vision Dict!")
        print("********")
        


        return self.conDict, self.CVCanvas, contourInfo_new


    def generateJsonDict(self, img, info, sizeOG):
        heightOG, widthOG, _ = sizeOG
        f,l = (widthOG*0.01,heightOG*0.01) if widthOG<heightOG else (heightOG*0.01,widthOG*0.01)

        height,width,_ = img.shape
        outerLayer = {"_EA@isEnabled": True, "_EA@class": "XCUIElementTypeOther", "_EA@isHidden": False, "_EA@isClickable": False, 
                    "tag": "ol", "_TaaD::byVision": True, "x": 0, "y": 0, "width": width, "height": height, "child":[]}

        secondLayerList = []
        copy = info.copy()
        while len(copy)>0:
            second = []
            c = copy[0]
            xc,yc,wc,hc = c[1]
            mid_yc = yc+(hc*0.5)
            second.append(c)
            for i,d in enumerate(copy[1:].copy()):
                xd,yd,wd,hd = d[1]
                mid_yd = yd+(hd*0.5)
                if abs(mid_yd-mid_yc)<=f*5 or (hd<=hc and wd<=wc and xd>=xc and xd+wd<=xc+wc and yd>=yc and yd+hd<=yc+hc):
                    for e in copy[i+1:].copy():
                        xe,ye,we,he = e[1]
                        mid_ye = ye+(he*0.5)
                        if abs(mid_ye-mid_yd)<=f*5 or (he<=hd and we<=wd and xe>=xd and xe+we<=xd+wd and ye>=yd and ye+he<=yd+hd):
                            if e in copy:
                                second.append(e)
                                copy.remove(e)
                    if d in copy:
                        second.append(d)
                        copy.remove(d)
                    
            copy.remove(c)
            secondLayerList.append(second)

        for second in secondLayerList:
            x_low, x_upp, y_low, y_upp = [width,0,height,0]
            for c in second:
                xc,yc,wc,hc = c[1]
                if xc<=x_low:
                    x_low = xc
                if yc<=y_low:
                    y_low = yc
                if xc+wc>=x_upp:
                    x_upp = xc+wc
                if yc+hc>=y_upp:
                    y_upp = yc+hc

            coordinate = [x_low, y_low, x_upp-x_low, y_upp-y_low]
            secondLayer = {"_EA@isEnabled": True, "_EA@class": "XCUIElementTypeOther", "_EA@isHidden": False, "_EA@isClickable": False, 
                        "tag": "li", "_TaaD::byVision": True, "x": coordinate[0], "y": coordinate[1], "width": coordinate[2], "height": coordinate[3], "child":[]}
            

            for c in second:
                #同層下一個輪廓的序號、同層上一個輪廓的序號、子輪廓的序號、父輪廓的序號。
                #[[輪廓編號, 輪廓hier資訊],[輪廓座標],[輪廓文字,confidence]]
                #因為子輪廓有複數個的話hier裡只會有第一個的編號，所以要用子輪廓中hier的父輪廓編號來判斷
                #所以不能用c[0][1][2]==c_2rd[0][0], 要用c[0][0]==c_2nd[0][1][3]
                #如果ocr confidence >= 80就加入xml dict?
                
                if c[0][1][3]==-1 or (c[0][1][3] != d[0][0] for d in info):
                    if c[2][1]>=70: 
                        inner_1st = self.addDict("XCUIElementTypeButton","button",c[2][0],c[1])
                    else: 
                        inner_1st = self.addDict("XCUIElementTypeButton","button","",c[1])
                        inner_1st.pop("_EA@text", None)
                        inner_1st["_TaaD::imageToBeClipped"] = True
                        
                    for i_2nd, c_2nd in enumerate(second):
                        
                        if c[0][0]==c_2nd[0][1][3]:
                            if c[2][1]>=70: 
                                inner_2nd = self.addDict("XCUIElementTypeButton","button",c_2nd[2][0],c_2nd[1])
                            else: 
                                inner_2nd = self.addDict("XCUIElementTypeButton","button","",c_2nd[1])
                                inner_2nd.pop("_EA@text", None)
                                inner_2nd["_TaaD::imageToBeClipped"] = True
                            
                            for i_3rd, c_3rd in enumerate(second):
                        
                                if c_2nd[0][0]==c_3rd[0][1][3]:
                                    if c[2][1]>=70: 
                                        inner_3rd= self.addDict("XCUIElementTypeButton","button",c_3rd[2][0],c_3rd[1])
                                    else: 
                                        inner_3rd = self.addDict("XCUIElementTypeButton","button","",c_3rd[1])
                                        inner_3rd.pop("_EA@text", None)
                                        inner_3rd["_TaaD::imageToBeClipped"] = True
                                    
                                    for i_4th, c_4th in enumerate(second):
                                        
                                        if c_3rd[0][0]==c_4th[0][1][3]:
                                            if c[2][1]>=70: 
                                                inner_4th = self.addDict("XCUIElementTypeButton","button",c_4th[2][0],c_4th[1])
                                            else: 
                                                inner_4th = self.addDict("XCUIElementTypeButton","button","",c_4th[1])
                                                inner_4th.pop("_EA@text", None)
                                                inner_4th["_TaaD::imageToBeClipped"] = True
                                            
                                            inner_3rd["child"].append(inner_4th)
                                            
                                    inner_2nd["child"].append(inner_3rd)
                            
                            inner_1st["child"].append(inner_2nd)
                    secondLayer['child'].append(inner_1st)

            outerLayer['child'].append(secondLayer)
        
        return outerLayer

    def addDict(self, elementType, tag, label, coordinate):
        x,y,w,h = coordinate
        a = {"_EA@isEnabled": True, "_EA@class": elementType, "_EA@isHidden": False, "_EA@isClickable": True, 
             "tag": tag, "op": "click", "_TaaD::byVision": True, "x": x, "y": y, "width": w, "height": h, "_EA@text": label, "child":[]}
        return a

    def combineContour(self, contourInfo_new, sizeOG):
        height, width, _ = sizeOG
        f,l = (width*0.01,height*0.01) if width<height else (height*0.01,width*0.01)
        for c in contourInfo_new.copy():
            xc,yc,wc,hc = c[1]
            tag = 0
            for d in contourInfo_new.copy():
                xd,yd,wd,hd = d[1]

                if hd<hc and wd<wc and xd>xc and xd+wd<xc+wc and yd>yc and yd+hd<yc+hc:
                    if wd+hd<=f*7 or (wd/hd>=3 and hd<=f*3) or (hd/wd>=3 and wd<=f*3):
                        if d in contourInfo_new:
                            contourInfo_new.remove(d)
                            #remove child that is too small and will be removed after combination
                    else: tag = 1
                if tag ==1: break
                #tag==1 means there is something contained in this contour
            if (wc>=width*0.8 or hc>=height*0.8) and tag==1:
                contourInfo_new.remove(c)
            #filter out contour that is big and has child
        for c in contourInfo_new.copy():
            xc,yc,wc,hc = c[1]
            if (xc==0 or xc==width) and wc<=width*0.05:
                contourInfo_new.remove(c)
            if (yc==0 or yc==height) and hc<=height*0.05:
                contourInfo_new.remove(c)
            #filter out edge contour

        ##first combine in & out contours
        for i,c in enumerate(contourInfo_new):
            for j,d in enumerate(contourInfo_new[i+1:].copy(),i+1):
                #print(len(contourInfo_new[i+1:].copy()))
                x1,y1,w1,h1 = d[1]
                x2,y2,w2,h2 = c[1]
                ##using distance of middle point of contours
                mid_x1 = x1+(w1/2)
                mid_y1 = y1+(h1/2)
                mid_x2 = x2+(w2/2)
                mid_y2 = y2+(h2/2)
                center_dist = math.sqrt((mid_x1 - mid_x2)**2 + (mid_y1 - mid_y2)**2)
                ctl_dtl = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                cbl_dbl = math.sqrt((x1 - x2)**2 + ((y1+h1) - (y2+h2))**2)
                ##define constraints
                d_in_c = (ctl_dtl<=f*10 or cbl_dbl<=f*10 or center_dist<=f*15)\
                        and (w1+h1 <=f*23 or abs(h1/w1 - h2/w2)<=0.2 or w2-w1<=3*f or h2-h1<=3*f)

                c_in_d = (ctl_dtl<=f*10 or cbl_dbl<=f*10 or center_dist<=f*15)\
                        and (w2+h2 <=f*23 or abs(h1/w1 - h2/w2)<=0.2 or w1-w2<=3*f or h1-h2<=3*f)

                temp=[]
                #print(i,j)
                ##c contains d and vice versa
                if h1<=h2 and w1<=w2 and x1>=x2 and x1+w1<=x2+w2+f and y1>=y2 and y1+h1<=y2+h2+f:
                    if d_in_c:
                        #print('d in c')
                        temp.append(d[0])

                        width = w2
                        height = h2
                        temp.append([min(x1,x2),min(y1,y2),width,height])
                        
                        
                        temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                        #temp[0][1][3]= -1

                        if d in contourInfo_new: contourInfo_new.remove(d)
                        contourInfo_new[i] = temp
                        c = contourInfo_new[i]
                        #break
                    else:
                        c[0][1][3]=-1
                        #如果是別人的子contour，但卻不適合和父合併，就拉到第一層

                elif h1>=h2 and w1>=w2 and x1<=x2 and x1+w1+f>=x2+w2 and y1<y2 and y1+h1+f>=y2+h2:
                    if c_in_d:
                        #print('c in d')
                        temp.append(d[0])

                        width = w1
                        height = h1
                        temp.append([min(x1,x2),min(y1,y2),width,height])
                        temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                        #temp[0][1][3]= -1

                        if d in contourInfo_new: contourInfo_new.remove(d)
                        contourInfo_new[i] = temp
                        c = contourInfo_new[i]
                        #break
                    else:
                        c[0][1][3]=-1
                        #如果是別人的子contour，但卻不適合和父合併，就拉到第一層
        
        for i,c in enumerate(contourInfo_new):
            last_step = ''
            for j,d in enumerate(contourInfo_new[i+1:].copy(),i+1):
                #print(len(contourInfo_new[i+1:].copy()))
                x1,y1,w1,h1 = d[1]
                x2,y2,w2,h2 = c[1]
                ##using distance of middle point of contours
                mid_x1 = x1+(w1/2)
                mid_y1 = y1+(h1/2)
                mid_x2 = x2+(w2/2)
                mid_y2 = y2+(h2/2)

                dtr_ctl = math.sqrt((x2 - (x1+w1))**2 + (y1 - y2)**2)
                dbr_cbl = math.sqrt((x2 - (x1+w1))**2 + ((y1+h1) - (y2+h2))**2)
                ctr_dtl = math.sqrt(((x2+w2) - x1)**2 + (y1 - y2)**2)
                cbr_dbl = math.sqrt(((x2+w2) - x1)**2 + ((y1+h1) - (y2+h2))**2)

                dbl_ctl = math.sqrt((x1 - x2)**2 + (y2 - (y1+h1))**2)
                dbr_ctr = math.sqrt(((x1+w1) - (x2+w2))**2 + (y2 - (y1+h1))**2)
                cbl_dtl = math.sqrt((x1 - x2)**2 + (y1 - (y2+h2))**2)
                cbr_dtr = math.sqrt(((x1+w1) - (x2+w2))**2 + (y1 - (y2+h2))**2)

                ##define constraints
                left =  (dtr_ctl<=f*4 or dbr_cbl<=f*4 or (x2 - (x1+w1))<=f*3.5 or x2<x1+w1) \
                        and not((abs(h1/w1 - h2/w2)<=0.05 or (abs(h1/w1 - h2/w2)<=0.15 and h1+w1<=15*f) or (0.7<=h1/h2<=1.4 and abs(w1-w2)<=f)) and abs(h1+w1-(h2+w2))<=6.5*f and (h1/w1<=3 and w1/h1<=3) and (x2 - (x1+w1))>=f*0.5) \
                        and (abs(h1-h2)<=f*20) and ((h1<f*25) or (h2<f*25)) \
                        and (abs(mid_y1-mid_y2)<=f*5 or (abs(y1-y2)<=f and abs(mid_y1-mid_y2)<=f*10) or (abs(y1+h1-(y2+h2))<=f and abs(mid_y1-mid_y2)<=f*10)) \
                        and not(h1>10*f and abs(h1-h2)<0.5*f)

                right = (ctr_dtl<=f*4 or cbr_dbl<=f*4 or (x1 - (x2+w2))<=f*3.5 or x1<x2+w2) \
                        and not((abs(h1/w1 - h2/w2)<=0.05 or (abs(h1/w1 - h2/w2)<=0.15 and h1+w1<=15*f) or (0.7<=h1/h2<=1.4 and abs(w1-w2)<=f)) and abs(h1+w1-(h2+w2))<=6.5*f and (h1/w1<=3 and w1/h1<=3) and (x1 - (x2+w2))>=f*0.5) \
                        and (abs(h1-h2)<=f*20) and ((h1<f*25) or (h2<f*25)) \
                        and (abs(mid_y1-mid_y2)<=f*5 or (abs(y1-y2)<=f and abs(mid_y1-mid_y2)<=f*10) or (abs(y1+h1-(y2+h2))<=f and abs(mid_y1-mid_y2)<=f*10)) \
                        and not(h1>10*f and abs(h1-h2)<0.5*f)
                
                top  = (dbl_ctl<=f*3.5 or dbr_ctr<=f*3.5 or ((w1<=w2 or abs(mid_x1-mid_x2)<=f) and (y2 - (y1+h1))<=f*3) or y2<y1+h1) \
                        and not((abs(h1/w1 - h2/w2)<=0.05 or (abs(h1/w1 - h2/w2)<=0.15 and h1+w1<=15*f) or (0.55<=w1/w2<=1.8 and abs(h1-h2)<=f)) and abs(h1+w1-(h2+w2))<=6.5*f and (h1/w1<=3 and w1/h1<=3) and (y2 - (y1+h1))>=f*0.5 and h1>=f*3 and w1>=f*3) \
                        and (abs(w1-w2)<f*22) and ((w1<f*21) or (w2<f*21)) \
                        and (abs(mid_x1-mid_x2)<=f*5 or (abs(x1-x2)<=f and abs(mid_x1-mid_x2)<=f*10) or (abs(x1+w1-(x2+h2))<=f and abs(mid_x1-mid_x2)<=f*10)) 
                
                bottom = (cbl_dtl<=f*3.5 or cbr_dtr<=f*3.5 or ((w2<=w1 or abs(mid_x1-mid_x2)<=f) and (y1 - (y2+h2))<=f*3) or y1<y2+h2) \
                        and not((abs(h1/w1 - h2/w2)<=0.05 or (abs(h1/w1 - h2/w2)<=0.15 and h1+w1<=15*f) or (0.55<=w1/w2<=1.8 and abs(h1-h2)<=f)) and abs(h1+w1-(h2+w2))<=6.5*f and (h1/w1<=3 and w1/h1<=3) and (y1 - (y2+h2))>=f*0.5 and h1>=f*3 and w1>=f*3) \
                        and (abs(w1-w2)<f*22) and ((w1<f*21) or (w2<f*21)) \
                        and (abs(mid_x1-mid_x2)<=f*5 or (abs(x1-x2)<=f and abs(mid_x1-mid_x2)<=f*10) or (abs(x1+w1-(x2+h2))<=f and abs(mid_x1-mid_x2)<=f*10)) 

                temp=[]
                #print(i,j)
                ##d is on left of c
                if x1<x2 and x1+w1<x2+w2 and (abs(mid_y1-mid_y2)<mid_x2-mid_x1):
                    if left:
                        #print("left")
                        if(last_step == 'bottom' or last_step =='top'): break
                        last_step = 'left'
                        temp.append(d[0])

                        if x1==x2:
                            width = max(w1,w2)
                        elif x1>x2 :
                            width = x1+w1-x2
                        else:
                            width = x2+w2-x1

                        if y1==y2:
                            height = max(h1,h2)
                        elif y1>y2 :
                            height = max(y1+h1,y2+h2)-y2
                        else:
                            height = max(y1+h1,y2+h2)-y1


                        temp.append([min(x1,x2),min(y1,y2),width,height])
                        if c[2][1]>=88 and d[2][1]>=88:
                            temp.append([c[2][0]+' '+d[2][0], c[2][1]])
                        else:
                            temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                        #左右關係的兩邊文字要相連
                        #temp[0][1][3]= -1

                        if d in contourInfo_new: contourInfo_new.remove(d)
                        contourInfo_new[i] = temp
                        c = contourInfo_new[i]
                        

                ##d is on right of c  
                elif x1>x2 and x1+w1>x2+w2 and (abs(mid_y1-mid_y2)<mid_x1-mid_x2):
                    if right:
                        #print("right")
                        if(last_step == 'bottom' or last_step =='top'): break
                        last_step = 'right'
                        temp.append(d[0])

                        if x1==x2:
                            width = max(w1,w2)
                        elif x1>x2 :
                            width = x1+w1-x2
                        else:
                            width = x2+w2-x1

                        if y1==y2:
                            height = max(h1,h2)
                        elif y1>y2 :
                            height = max(y1+h1,y2+h2)-y2
                        else:
                            height = max(y1+h1,y2+h2)-y1


                        temp.append([min(x1,x2),min(y1,y2),width,height])
                        if c[2][1]>=88 and d[2][1]>=88:
                            temp.append([c[2][0]+' '+d[2][0], c[2][1]])
                        else:
                            temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                        #左右關係的兩邊文字要相連
                        #temp[0][1][3]= -1

                        if d in contourInfo_new: contourInfo_new.remove(d)
                        contourInfo_new[i] = temp
                        c = contourInfo_new[i]

                #d is on top of c
                elif y1<y2 and y1+h1<y2+h2 and (abs(mid_x1-mid_x2)< mid_y2-mid_y1):
                    if top:
                        #print("top")
                        if(last_step == 'left' or last_step =='right'): break
                        last_step = 'top'
                        temp.append(d[0])
                        

                        if x1==x2:
                            width = max(w1,w2)
                        elif x1>x2:
                            width = max(x1+w1,x2+w2)-x2
                        else:
                            width = max(x1+w1,x2+w2)-x1

                        if y1==y2:
                            height = max(h1,h2)
                        elif y1>y2 :
                            height = y1+h1-y2
                        else:
                            height = y2+h2-y1


                        temp.append([min(x1,x2),min(y1,y2),width,height])
                        temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                        #temp[0][1][3]= -1

                        if d in contourInfo_new: contourInfo_new.remove(d)
                        contourInfo_new[i] = temp
                        c = contourInfo_new[i]
                        

                #d is at the bottom of c
                elif y1>y2 and y1+h1>y2+h2 and (abs(mid_x1-mid_x2)< mid_y1-mid_y2):
                    if bottom:
                        #print("bottom")
                        if(last_step == 'left' or last_step =='right'): break
                        last_step = 'bottom'
                        temp.append(d[0])

                        if x1==x2:
                            width = max(w1,w2)
                        elif x1>x2:
                            width = max(x1+w1,x2+w2)-x2
                        else:
                            width = max(x1+w1,x2+w2)-x1



                        if y1==y2:
                            height = max(h1,h2)
                        elif y1>y2 :
                            height = y1+h1-y2
                        else:
                            height = y2+h2-y1


                        temp.append([min(x1,x2),min(y1,y2),width,height])
                        temp.append(c[2] if c[2][1]>d[2][1] else d[2])

                        #temp[0][1][3]= -1

                        if d in contourInfo_new: contourInfo_new.remove(d)
                        contourInfo_new[i] = temp
                        c = contourInfo_new[i]

        for  c in contourInfo_new.copy():
            x,y,w,h = c[1]
            if w+h<=f*7 or (w/h>=3 and h<=f*3) or (h/w>=3 and w<=f*3):
                contourInfo_new.remove(c)
            #filter out elements that too small

        return contourInfo_new

    def removeOverlap(self, contourInfo_new, sizeOG):
        height, width, _ = sizeOG
        f,l = (width*0.01,height*0.01) if width<height else (height*0.01,width*0.01)

        ##first filter out contained contours
        for i,c in enumerate(contourInfo_new):
            for j,d in enumerate(contourInfo_new[i+1:].copy(),i+1):
                #print(len(contourInfo_new[i+1:].copy()))
                x1,y1,w1,h1 = d[1]
                x2,y2,w2,h2 = c[1]
                temp=[]
                #print(i,j)
                ##c contains d and vice versa
                if h1<=h2 and w1<=w2 and x1>=x2 and x1+w1<=x2+w2+f and y1>=y2 and y1+h1<=y2+h2+f:
                    temp.append(d[0])

                    width = w2
                    height = h2
                    temp.append([min(x1,x2),min(y1,y2),width,height])
                    
                    
                    temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                    #temp[0][1][3]= -1

                    if d in contourInfo_new: contourInfo_new.remove(d)
                    contourInfo_new[i] = temp
                    c = contourInfo_new[i]
                    #if d in contourInfo_new: contourInfo_new.remove(d)

                elif h1>=h2 and w1>=w2 and x1<=x2 and x1+w1+f>=x2+w2 and y1<=y2 and y1+h1+f>=y2+h2:
                    temp.append(d[0])

                    width = w1
                    height = h1
                    temp.append([min(x1,x2),min(y1,y2),width,height])
                    temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                    #temp[0][1][3]= -1

                    if d in contourInfo_new: contourInfo_new.remove(d)
                    contourInfo_new[i] = temp
                    c = contourInfo_new[i]
                    #if c in contourInfo_new: contourInfo_new.remove(c)
            
        for i,c in enumerate(contourInfo_new):
            for j,d in enumerate(contourInfo_new[i+1:].copy(),i+1):
                #print(len(contourInfo_new[i+1:].copy()))
                x1,y1,w1,h1 = d[1]
                x2,y2,w2,h2 = c[1]
                temp=[]
                ##d is on left of c
                if x1<x2 and x1+w1<x2+w2 and x2<x1+w1 \
                and (y1<=y2<=y1+h1 or y1<=y2+h2<=y1+h1 or y1<=y2<=y1+h1 or (y2>=y1 and y2+h2<=y1+h1) or (y1>=y2 and y1+h1<=y2+h2)):
                    #print("left")
                    temp.append(d[0])

                    if x1==x2:
                        width = max(w1,w2)
                    elif x1>x2 :
                        width = x1+w1-x2
                    else:
                        width = x2+w2-x1

                    if y1==y2:
                        height = max(h1,h2)
                    elif y1>y2 :
                        height = max(y1+h1,y2+h2)-y2
                    else:
                        height = max(y1+h1,y2+h2)-y1


                    temp.append([min(x1,x2),min(y1,y2),width,height])
                    if c[2][1]>=88 and d[2][1]>=88:
                        temp.append([d[2][0]+' '+c[2][0], c[2][1]])
                    else:
                        temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                    #左右關係的兩邊文字要相連
                    #temp[0][1][3]= -1

                    if d in contourInfo_new: contourInfo_new.remove(d)
                    contourInfo_new[i] = temp
                    c = contourInfo_new[i]
                    

                ##d is on right of c  
                elif x1>x2 and x1+w1>x2+w2 and x1<x2+w2 \
                    and (y1<=y2<=y1+h1 or y1<=y2+h2<=y1+h1 or (y2>=y1 and y2+h2<=y1+h1) or (y1>=y2 and y1+h1<=y2+h2)):

                    #print("right")
                    temp.append(d[0])

                    if x1==x2:
                        width = max(w1,w2)
                    elif x1>x2 :
                        width = x1+w1-x2
                    else:
                        width = x2+w2-x1

                    if y1==y2:
                        height = max(h1,h2)
                    elif y1>y2 :
                        height = max(y1+h1,y2+h2)-y2
                    else:
                        height = max(y1+h1,y2+h2)-y1


                    temp.append([min(x1,x2),min(y1,y2),width,height])
                    if c[2][1]>=88 and d[2][1]>=88:
                        temp.append([c[2][0]+' '+d[2][0], c[2][1]])
                    else:
                        temp.append(c[2] if c[2][1]>d[2][1] else d[2])
                    #左右關係的兩邊文字要相連
                    #temp[0][1][3]= -1

                    if d in contourInfo_new: contourInfo_new.remove(d)
                    contourInfo_new[i] = temp
                    c = contourInfo_new[i]

                #d is on top of c
                elif y1<y2 and y1+h1<y2+h2 and y2<y1+h1 \
                    and (x1<=x2<=x1+w1 or x1<=x2+w2<=x1+w1 or (x1>=x2 and x1+w1<=x2+w2) or (x2>=x1 and x2+w2<=x1+w1)):
                    #print("top")
                    temp.append(d[0])
                    

                    if x1==x2:
                        width = max(w1,w2)
                    elif x1>x2:
                        width = max(x1+w1,x2+w2)-x2
                    else:
                        width = max(x1+w1,x2+w2)-x1

                    if y1==y2:
                        height = max(h1,h2)
                    elif y1>y2 :
                        height = y1+h1-y2
                    else:
                        height = y2+h2-y1


                    temp.append([min(x1,x2),min(y1,y2),width,height])
                    if c[2][1]>=88 and d[2][1]>=88:
                        temp.append([d[2][0]+' '+c[2][0], c[2][1]])
                    else:
                        temp.append(c[2] if c[2][1]>d[2][1] else d[2])

                    if d in contourInfo_new: contourInfo_new.remove(d)
                    contourInfo_new[i] = temp
                    c = contourInfo_new[i]
                        

                #d is at the bottom of c
                elif y1>y2 and y1+h1>y2+h2 and y1<y2+h2 \
                    and (x1<=x2<=x1+w1 or x1<=x2+w2<=x1+w1 or (x1>=x2 and x1+w1<=x2+w2) or (x2>=x1 and x2+w2<=x1+w1)):
                    #print("bottom")
                    temp.append(d[0])

                    if x1==x2:
                        width = max(w1,w2)
                    elif x1>x2:
                        width = max(x1+w1,x2+w2)-x2
                    else:
                        width = max(x1+w1,x2+w2)-x1



                    if y1==y2:
                        height = max(h1,h2)
                    elif y1>y2 :
                        height = y1+h1-y2
                    else:
                        height = y2+h2-y1


                    temp.append([min(x1,x2),min(y1,y2),width,height])
                    if c[2][1]>=88 and d[2][1]>=88:
                        temp.append([c[2][0]+' '+d[2][0], c[2][1]])
                    else:
                        temp.append(c[2] if c[2][1]>d[2][1] else d[2])

                    if d in contourInfo_new: contourInfo_new.remove(d)
                    contourInfo_new[i] = temp
                    c = contourInfo_new[i]

        return contourInfo_new
    def addText(self, c, img_gray):
        x,y,w,h = c[1]
        flag=0
        for j in range(1):
            ratio = [h/(h+w), w/(h+w)]
            s = 0.1*j*w if w>h else 0.1*j*h
            ratio = [int(s*k) for k in ratio]
            y_low = y-ratio[0] if y-ratio[0]>=0 else 0
            y_upp = y+h+ratio[0] if y+h+ratio[0]<img_gray.shape[0] else img_gray.shape[0]
            x_low = x-ratio[1] if x-ratio[1]>=0 else 0
            x_upp = x+w+ratio[1] if x+w+ratio[1]<img_gray.shape[1] else img_gray.shape[1]
            img_crop = img_gray[y_low:y_upp, x_low:x_upp].copy()
            hc,wc = img_crop.shape
            if j==0: mean_crop = np.average(img_crop)
            if mean_crop <= 140:
                ret,img_crop = cv2.threshold(img_crop,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            else:
                ret,img_crop = cv2.threshold(img_crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ##binarize
                
            if y_low!=0: 
                img_crop[0:ratio[0],:] = 255
                img_crop[ratio[0]+h:hc,:] = 255
            else: 
                img_crop[0:y,:] = 255
                img_crop[y+h:hc,:] = 255
                
            if x_low!=0: 
                img_crop[:,0:ratio[1]] = 255
                img_crop[:,ratio[1]+w:wc] = 255
            else: 
                img_crop[:,0:x] = 255
                img_crop[:,x+w:wc] = 255
            ##zoom out and set border pixels to 255

            ocrResult, conf = self.getOcrResult(img_crop)

            if conf==0: flag+=1 
            else: flag=0
            if conf<=70:
                kernel = np.ones((3,3), np.uint8)
                img_crop_ero = cv2.erode(img_crop, kernel, iterations = 1)
                ocrResult1,conf1 = self.getOcrResult(img_crop_ero)


                img_crop_dil = cv2.dilate(img_crop, kernel, iterations = 1)
                ocrResult2,conf2 = self.getOcrResult(img_crop_dil)


                if conf1>=conf2:
                    ocrResult = ocrResult1
                    conf = conf1
                else:
                    ocrResult = ocrResult2
                    conf = conf2
                    ##conf<70 then dilate or erode

                if conf==0: flag+=1 
                else: flag=0
            
            if conf>=95 or (conf>=80 and j>=3) or (conf>=70 and j>=10) or flag==4: break
        
        c.append([ocrResult,conf])


    def addTextInfo(self, contourInfo, img_gray):
        for i, c in enumerate(contourInfo):
            x,y,w,h = c[1]
            flag=0
            for j in range(1):
                ratio = [h/(h+w), w/(h+w)]
                s = 0.1*j*w if w>h else 0.1*j*h
                ratio = [int(s*k) for k in ratio]
                y_low = y-ratio[0] if y-ratio[0]>=0 else 0
                y_upp = y+h+ratio[0] if y+h+ratio[0]<img_gray.shape[0] else img_gray.shape[0]
                x_low = x-ratio[1] if x-ratio[1]>=0 else 0
                x_upp = x+w+ratio[1] if x+w+ratio[1]<img_gray.shape[1] else img_gray.shape[1]
                img_crop = img_gray[y_low:y_upp, x_low:x_upp].copy()
                hc,wc = img_crop.shape
                if j==0: mean_crop = np.average(img_crop)
                if mean_crop <= 140:
                    ret,img_crop = cv2.threshold(img_crop,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                else:
                    ret,img_crop = cv2.threshold(img_crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                ##binarize
                    
                if y_low!=0: 
                    img_crop[0:ratio[0],:] = 255
                    img_crop[ratio[0]+h:hc,:] = 255
                else: 
                    img_crop[0:y,:] = 255
                    img_crop[y+h:hc,:] = 255
                    
                if x_low!=0: 
                    img_crop[:,0:ratio[1]] = 255
                    img_crop[:,ratio[1]+w:wc] = 255
                else: 
                    img_crop[:,0:x] = 255
                    img_crop[:,x+w:wc] = 255
                ##zoom out and set border pixels to 255

                ocrResult, conf = self.getOcrResult(img_crop)

                if conf==0: flag+=1 
                else: flag=0
                if conf<=70:
                    kernel = np.ones((3,3), np.uint8)
                    img_crop_ero = cv2.erode(img_crop, kernel, iterations = 1)
                    ocrResult1,conf1 = self.getOcrResult(img_crop_ero)


                    img_crop_dil = cv2.dilate(img_crop, kernel, iterations = 1)
                    ocrResult2,conf2 = self.getOcrResult(img_crop_dil)


                    if conf1>=conf2:
                        ocrResult = ocrResult1
                        conf = conf1
                    else:
                        ocrResult = ocrResult2
                        conf = conf2
                        ##conf<70 then dilate or erode

                    if conf==0: flag+=1 
                    else: flag=0
                
                if conf>=95 or (conf>=80 and j>=3) or (conf>=70 and j>=10) or flag==4: break
            
            c.append([ocrResult,conf])


        return contourInfo
        


    def getOcrResult(self, img):
        ##tesserocr method with threads
        api = None
        try:
            api = self.tesserocr_queue.get(block=True, timeout=300)
            #print('Api Acquired')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            pil = Image.fromarray(img)
            api.SetImage(pil)
            ocrResult = api.GetUTF8Text()
            conf = api.MeanTextConf()
            ocrResult = ocrResult.strip().replace(" ","").replace("\n"," ")
        except queue.Empty:
            print('Empty exception caught!')
            return None
        finally:
            if api is not None:
                self.tesserocr_queue.put(api)
                #print('Api released')

        ##pytesseract method
        # ocr = pytesseract.image_to_data(img,lang=self.langs, config=self.ocr_config, output_type='data.frame')
        # if True in list(ocr.conf>=70):
        #     text = ocr[ocr.conf>=70]
        #     #lines = text.groupby('block_num')['text'].apply(list)[1]
        #     #conf = text.groupby(['block_num'])['conf'].mean()[1]
        #     lines = list(text.text)
        #     conf = round(sum(list(text.conf)) / len(list(text.conf)))
        #     ocrResult = ''.join(str(i) for i in lines)
        # else:
        #     ocrResult = ''
        #     conf = 0
        
        
        return ocrResult,conf

    def getContourInfo(self, gray_edges, sizeOG):
        height,width, _ = sizeOG
        long = height if height>width else width
        short = width if long == height else height
        (result, hierarchy) = cv2.findContours(gray_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print("number of contour result:",np.size(result))

        new = []
        hier = []
        size = []
        contourInfo = []
        for a,b,c in zip(result, hierarchy[0].tolist(), range(len(result))):
            
            (x, y, w, h) = cv2.boundingRect(a)
            
            if b[3] == -1:
                new.append(a)
                hier.append([c,b])

            elif b[2] != -1:
                new.append(a)
                hier.append([c,b])
            elif w >= short/15 and h>=short/15 and np.size(result)<250:
                new.append(a)
                hier.append([c,b])
                
        for c in new:
            (x, y, w, h) = cv2.boundingRect(c)
            size.append([x,y,w,h])
        
        print("number of contours after filted by hierarchy:",len(hier))
        contourInfo = [list(item) for item in zip(hier,size)]
        
        for c in contourInfo.copy():
            x,y,w,h = c[1]
            if (w/h<=0.1 and h<=height/2) or w/h<=0.05 or (h/w<=0.1 and w<=width/2) or h/w<=0.05:
                contourInfo.remove(c)

                #去掉直長條或橫長條
        print('number of contours after filted by size:', len(contourInfo))

        return contourInfo
    def removeImgNoise(self, contourInfo, img_gray, sizeOG):
        temp = contourInfo.copy()
        pixelInfo = []
        h,w,_  = sizeOG
        dist_factor = w*0.01 if w<h else h*0.01
        for c in temp:
            x0,y0,w0,h0 = c[1]
            img_crop = img_gray[y0:y0+h0, x0:x0+w0].copy()
            unique, counts = np.unique(img_crop, return_counts=True)
            pixelInfo.append([max(counts), img_crop.size])
            
        for c,e in zip(temp, pixelInfo):
            xc,yc,wc,hc = c[1]
            maxc, sizec = e
            if (maxc/sizec<=0.43 and hc<=0.95*h and wc<=0.95*w) or maxc/sizec<=0.24:
                for d, f in zip(temp, pixelInfo):

                    xd,yd,wd,hd = d[1]
                    maxd, sized = f
                    if hd<hc and wd<wc and xd>=xc and xd+wd<=xc+wc and yd>=yc and yd+hd<=yc+hc:
                        if maxd/sized<=0.43 and wc-wd>=3*dist_factor and hc-hd>=3*dist_factor:
                            if d in contourInfo: contourInfo.remove(d)
        print('number of contours after filted out img noises:', len(contourInfo))
        return contourInfo
    
    def addDilate(self, imgData, size):
        dilationSize = size 
        kernel = np.ones((2 * dilationSize + 1, 2 * dilationSize + 1), np.uint8)
        img_dilation = cv2.dilate(imgData, kernel, iterations=1)
        return img_dilation

    def updateLowHighThreshold(self, imgData):
        imgHist = cv2.equalizeHist(imgData)
        mean = np.average(imgHist)
        lowThreshold = int(self.CANNY_LOW_THRESHOLD_RATIO * mean * self.CANNY_RATIO_CONTROL_THRESHOLD)
        highThreshold = int(self.CANNY_LOW_THRESHOLD_RATIO * self.ratio * mean * self.CANNY_RATIO_CONTROL_THRESHOLD)
        
        return (lowThreshold, highThreshold, mean)
