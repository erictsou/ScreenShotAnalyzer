
##辨識截圖上的文字並且加入contourInfo

#import tesserocr
#import pytesseract
import cv2
from PIL import Image
import queue
import numpy as np

tesserocr_queue = queue.Queue()

def getOcrResult(img):
    
    ## tesserocr method
    api = None
    try:
        t_num, api = tesserocr_queue.get(block=True, timeout=300)
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
            tesserocr_queue.put([t_num,api])
            #print('Api released')
    
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     pil = Image.fromarray(img)
#     api.SetImage(pil)
#     ocrResult = api.GetUTF8Text()
#     conf = api.MeanTextConf()
#     ocrResult = ocrResult.strip().replace(" ","").replace("\n"," ")

    
    ##pytesseract method
#     ocr = pytesseract.image_to_data(img,lang='eng+chi_tra',config='--psm 6', output_type='data.frame')
    
#     if True in list(ocr.conf>=70):
#         #print(ocr)
#         text = ocr[ocr.conf!=-1]
#         #lines = text.groupby('block_num')['text'].apply(list)[1]
#         #conf = text.groupby(['block_num'])['conf'].mean()[1]
#         lines = list(text.text)
#         conf = round(sum(list(text.conf)) / len(list(text.conf)))
#         ocrResult = ''.join(str(i) for i in lines)
#     else:
#         ocrResult = ''
#         conf = 0
    
    
    return ocrResult,conf,t_num



#for i, c in enumerate(contourInfo):
def addText(c, img_gray):
    x,y,w,h = c[1]
#     x = int(x+0.02*w)
#     y = int(y+0.05*h)
#     w = int(0.96*w)
#     h = int(0.9*h)
    #api.SetRectangle(x, y, w, h)
    ##用tesseract的setRectangle來切割
    i=0
    flag=0
    for j in range(1):
        ratio = [h/(h+w), w/(h+w)]
        s = 0.1*j*w if w>h else 0.1*j*h
        ratio = [int(s*k) for k in ratio]
        ##要設定s還是固定距離就好？？
        y_low = y-ratio[0] if y-ratio[0]>=0 else 0
        y_upp = y+h+ratio[0] if y+h+ratio[0]<img_gray.shape[0] else img_gray.shape[0]
        x_low = x-ratio[1] if x-ratio[1]>=0 else 0
        x_upp = x+w+ratio[1] if x+w+ratio[1]<img_gray.shape[1] else img_gray.shape[1]
        #img_crop = img_gray[y-ratio[0]:y+h+ratio[0], x-ratio[1]:x+w+ratio[1]].copy()
        img_crop = img_gray[y_low:y_upp, x_low:x_upp].copy()
        #unique, counts = np.unique(img_crop, return_counts=True)

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
        ##等比例拉遠並把外圍pixel設為255

        #img_crop = cv2.resize(img_crop, (int(wc*10),int(hc*10)))
        #圖片放大
                
        ocrResult, conf,t_num = getOcrResult(img_crop)
        ##直接切割圖片後各自辨識
            

        
        if conf==0: flag+=1 
        else: flag=0
        if conf<=70:
            kernel = np.ones((3,3), np.uint8)
            img_crop_ero = cv2.erode(img_crop, kernel, iterations = 1)
            ocrResult1,conf1,t_num = getOcrResult(img_crop_ero)


            img_crop_dil = cv2.dilate(img_crop, kernel, iterations = 1)
            ocrResult2,conf2,t_num = getOcrResult(img_crop_dil)


            if conf1>=conf2:
                ocrResult = ocrResult1
                conf = conf1
                print('get text thicker')
            else:
                ocrResult = ocrResult2
                conf = conf2
                print('get text thinner')
                ##辨識不出來就加粗字體or變細

            if conf==0: flag+=1 
            else: flag=0
        
        if conf>=95 or (conf>=80 and j>=3) or (conf>=70 and j>=10) or flag==4: break
    
    print("round[{7}]Box[{0}]thread[{8}]: x={3}, y={4}, w={5}, h={6}, confidence: {1}, text: {2}".format(i, conf, ocrResult, x,y,w,h,j,t_num),c[0])
    
    #if i==5:break
    c.append([ocrResult,conf])
    
    #c.append([max(counts), img_crop.size, len(unique)])




## threading version
# def getOcrResult(api,img):
    
#     ## tesserocr method
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     pil = Image.fromarray(img)
#     api.SetImage(pil)
#     ocrResult = api.GetUTF8Text()
#     conf = api.MeanTextConf()
#     ocrResult = ocrResult.strip().replace(" ","").replace("\n"," ")

    
#     ##pytesseract method
# #     ocr = pytesseract.image_to_data(img,lang='eng+chi_tra',config='--psm 6', output_type='data.frame')
    
# #     if True in list(ocr.conf>=70):
# #         #print(ocr)
# #         text = ocr[ocr.conf!=-1]
# #         #lines = text.groupby('block_num')['text'].apply(list)[1]
# #         #conf = text.groupby(['block_num'])['conf'].mean()[1]
# #         lines = list(text.text)
# #         conf = round(sum(list(text.conf)) / len(list(text.conf)))
# #         ocrResult = ''.join(str(i) for i in lines)
# #     else:
# #         ocrResult = ''
# #         conf = 0
    
    
#     return ocrResult,conf

# def addText(contourInfo, img_gray):
#     api = None
#     try:
#         t_num, api = tesserocr_queue.get(block=True, timeout=300)
#         print('Api Acquired')
#     except queue.Empty:
#         print('Empty exception caught!')
#         return None
    
#     for i, c in enumerate(contourInfo):
#         x,y,w,h = c[1]
#         ##用tesseract的setRectangle來切割

#         flag=0
#         for j in range(1):
#             ratio = [h/(h+w), w/(h+w)]
#             s = 0.1*j*w if w>h else 0.1*j*h
#             ratio = [int(s*k) for k in ratio]
#             ##要設定s還是固定距離就好？？
#             y_low = y-ratio[0] if y-ratio[0]>=0 else 0
#             y_upp = y+h+ratio[0] if y+h+ratio[0]<img_gray.shape[0] else img_gray.shape[0]
#             x_low = x-ratio[1] if x-ratio[1]>=0 else 0
#             x_upp = x+w+ratio[1] if x+w+ratio[1]<img_gray.shape[1] else img_gray.shape[1]
#             #img_crop = img_gray[y-ratio[0]:y+h+ratio[0], x-ratio[1]:x+w+ratio[1]].copy()
#             img_crop = img_gray[y_low:y_upp, x_low:x_upp].copy()
#             hc,wc = img_crop.shape
#             if j==0: mean_crop = np.average(img_crop)
#             if mean_crop <= 140:
#                 ret,img_crop = cv2.threshold(img_crop,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#             else:
#                 ret,img_crop = cv2.threshold(img_crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#             ##binarize

#             if y_low!=0: 
#                 img_crop[0:ratio[0],:] = 255
#                 img_crop[ratio[0]+h:hc,:] = 255
#             else: 
#                 img_crop[0:y,:] = 255
#                 img_crop[y+h:hc,:] = 255

#             if x_low!=0: 
#                 img_crop[:,0:ratio[1]] = 255
#                 img_crop[:,ratio[1]+w:wc] = 255
#             else: 
#                 img_crop[:,0:x] = 255
#                 img_crop[:,x+w:wc] = 255
#             ##等比例拉遠並把外圍pixel設為255

#             #img_crop = cv2.resize(img_crop, (int(wc*10),int(hc*10)))
#             #圖片放大

#             ocrResult, conf = getOcrResult(api,img_crop)
#             ##直接切割圖片後各自辨識



#             if conf==0: flag+=1 
#             else: flag=0
#             if conf<=70:
#                 kernel = np.ones((3,3), np.uint8)
#                 img_crop_ero = cv2.erode(img_crop, kernel, iterations = 1)
#                 ocrResult1,conf1 = getOcrResult(api,img_crop_ero)


#                 img_crop_dil = cv2.dilate(img_crop, kernel, iterations = 1)
#                 ocrResult2,conf2 = getOcrResult(api,img_crop_dil)


#                 if conf1>=conf2:
#                     ocrResult = ocrResult1
#                     conf = conf1
#                     print('get text thicker')
#                 else:
#                     ocrResult = ocrResult2
#                     conf = conf2
#                     print('get text thinner')
#                     ##辨識不出來就加粗字體or變細

#                 if conf==0: flag+=1 
#                 else: flag=0

#             if conf>=95 or (conf>=80 and j>=3) or (conf>=70 and j>=10) or flag==4: break

#         print("round[{7}]Box[{0}]thread[{8}]: x={3}, y={4}, w={5}, h={6}, confidence: {1}, text: {2}".format(i, conf, ocrResult, x,y,w,h,j,t_num),c[0])

#         #if i==5:break
#         c.append([ocrResult,conf])
#         cv2.rectangle(clone3, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(clone3, str(c[0][0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     if api is not None:
#         tesserocr_queue.put([t_num,api])
#         print('Api released')


        

# for t_num in range(NUM_THREADS):
#     tesserocr_queue.put([t_num, tesserocr.PyTessBaseAPI(lang='eng+chi_tra',psm=6)])

# start = time.time()

# contourThreadList = []
# part_num = int(len(contourInfo)/NUM_THREADS) +1
# cnt=0
# i=1
# while cnt<=len(contourInfo):
    
#     print(cnt, part_num*i)
#     contourThreadList.append(contourInfo[cnt:part_num*i])
    
#     cnt += part_num
#     i+=1
    
# print('num_thread:', len(contourThreadList))
    
# threads = []
# for i, c in enumerate(contourThreadList):
#     threads.append(threading.Thread(target = addText, args = (contourThreadList[i],img_gray)))
#     threads[i].start()

# # 主執行緒繼續執行自己的工作
# # ...

# # 等待所有子執行緒結束
# for i in range(NUM_THREADS):
#     threads[i].join()

# print("Done.")

# end = time.time()
# print('took tot: ' + str(end - start))
# for _ in range(NUM_THREADS):
#     t_num, api = tesserocr_queue.get(block=True)
#     api.End()
