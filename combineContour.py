import math 
#同層下一個輪廓的序號、同層上一個輪廓的序號、子輪廓的序號、父輪廓的序號。
#contourInfo: [[輪廓編號, 輪廓hier資訊],[x,y,w,h],[輪廓文字,confidence],[max(count),pixel size]]
def combineContour(contourInfo_new, sizeOG):
    height, width, _ = sizeOG
    f,l = (width*0.01,height*0.01) if width<height else (height*0.01,width*0.01)
    for c in contourInfo_new.copy():
        xc,yc,wc,hc = c[1]
        tag = 0
        for d in contourInfo_new.copy():
            xd,yd,wd,hd = d[1]

            if hd<hc and wd<wc and xd>xc and xd+wd<xc+wc and yd>yc and yd+hd<yc+hc:
                if wd+hd<=f*7 or (wd/hd>=3 and hd<=f*3) or (hd/wd>=3 and wd<=f*3):
                    contourInfo_new.remove(d)
                    print('remove small', d)
                    #remove child that is too small and will be removed after combination
                else: tag = 1
            if tag ==1: break
            #tag==1 means there is something contained in this contour
        if (wc>=width*0.8 or hc>=height*0.8) and tag==1:
            contourInfo_new.remove(c)
            print('remove', c)
        #filter out contour that is big and has child
    for c in contourInfo_new.copy():
        xc,yc,wc,hc = c[1]
        if (xc==0 or xc==width) and wc<=width*0.05:
            contourInfo_new.remove(c)
            print('remove edge', c)
        if (yc==0 or yc==height) and hc<=height*0.05:
            contourInfo_new.remove(c)
            print('remove edge', c)
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

            if c[0][0]==35:
                pass
                # print(i,c)
                # print(j,d)
                


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