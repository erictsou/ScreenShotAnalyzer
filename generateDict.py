import json
import cv2
# import lxml.etree as ET

def addDict(elementType, tag, label, coordinate):
    x,y,w,h = coordinate
    a = {"_EA@isEnabled": True, "_EA@class": elementType, "_EA@isHidden": False, "_EA@isClickable": True, 
        "tag": tag, "op": "click", "_TaaD::byVision": True, "x": x, "y": y, "width": w, "height": h, "_EA@text": label, "child":[]}
    return a
    
def generateJsonDict(filename, img, info, sizeOG, clone6):
    heightOG, widthOG, _ = sizeOG
    f,l = (widthOG*0.01,heightOG*0.01) if widthOG<heightOG else (heightOG*0.01,widthOG*0.01)

    height,width,_ = img.shape
    outerLayer = {"_EA@isEnabled": True, "_EA@class": "XCUIElementTypeOther", "_EA@isHidden": False, "_EA@isClickable": False, 
                "tag": "ol", "_TaaD::byVision": True, "x": 0, "y": 0, "width": width, "height": height, "child":[]}


    secondLayerList = []
    cnt2=0
    cnt=0
    temp = []
    copy = info.copy()
    while len(copy)>0:
        second = []
        c = copy[0]
        xc,yc,wc,hc = c[1]
        mid_yc = yc+(hc*0.5)
        second.append(c)
        cnt2+=1
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
                            cnt2+=1
                            copy.remove(e)
                if d in copy:
                    second.append(d)
                    cnt2+=1
                    copy.remove(d)
                
        copy.remove(c)
        secondLayerList.append(second)

    print(cnt2)
    #print(secondLayerList)

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
        cv2.rectangle(clone6, (x_low, y_low), (x_upp, y_upp), (0, 255, 0), 2)
        secondLayer = {"_EA@isEnabled": True, "_EA@class": "XCUIElementTypeOther", "_EA@isHidden": False, "_EA@isClickable": False, 
                    "tag": "li", "_TaaD::byVision": True, "x": coordinate[0], "y": coordinate[1], "width": coordinate[2], "height": coordinate[3], "child":[]}
        

        sym = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', \
            '<', '=', '>', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '©', '®']
        alpha_str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        alpha = list(alpha_str)
        ##remove abnormal symbol text
        for c in second:
            #同層下一個輪廓的序號、同層上一個輪廓的序號、子輪廓的序號、父輪廓的序號。
            #[[輪廓編號, 輪廓hier資訊],[輪廓座標],[輪廓文字,confidence]]
            #因為子輪廓有複數個的話hier裡只會有第一個的編號，所以要用子輪廓中hier的父輪廓編號來判斷
            #所以不能用c[0][1][2]==c_2rd[0][0], 要用c[0][0]==c_2nd[0][1][3]
            #如果ocr confidence >= 80就加入xml dict?
            
            if c[0][1][3]==-1 or (c[0][1][3] != d[0][0] for d in info):
                print(c[0][0],c[0][1],c[1])
                temp.append(c[1])
                cnt+=1
                if c[2][1]>=70 and \
                    not(not(False in (s in sym for s in c[2][0])) \
                    or (not(False in (s in alpha+sym for s in c[2][0])) and len(c[2][0])<=2) \
                    or len(c[2][0])<=1 or c[2][0]==len(c[2][0])*c[2][0][0]):

                    print('text: ',c[2])
                    inner_1st = addDict("XCUIElementTypeButton","button",c[2][0],c[1])
                else: 
                    inner_1st = addDict("XCUIElementTypeButton","button","",c[1])
                    inner_1st.pop("_EA@text", None)
                    inner_1st["_TaaD::imageToBeClipped"] = True
                    
                for i_2nd, c_2nd in enumerate(second):
                    
                    if c[0][0]==c_2nd[0][1][3]:
                        print(c_2nd[0][0],c_2nd[0][1],c_2nd[1],'2nd')
                        temp.append(c_2nd[1])
                        cnt+=1
                        if c[2][1]>=70 and \
                            not(not(False in (s in sym for s in c[2][0])) \
                            or (not(False in (s in alpha+sym for s in c[2][0])) and len(c[2][0])<=2) \
                            or len(c[2][0])<=1 or c[2][0]==len(c[2][0])*c[2][0][0]):

                            print('text: ',c_2nd[2])
                            inner_2nd = addDict("XCUIElementTypeButton","button",c_2nd[2][0],c_2nd[1])
                        else: 
                            inner_2nd = addDict("XCUIElementTypeButton","button","",c[1])
                            inner_2nd.pop("_EA@text", None)
                            inner_2nd["_TaaD::imageToBeClipped"] = True
                        
                        for i_3rd, c_3rd in enumerate(second):
                    
                            if c_2nd[0][0]==c_3rd[0][1][3]:
                                print(c_3rd[0][0],c_3rd[0][1],c_3rd[1],'3rd')
                                temp.append(c_3rd[1])
                                cnt+=1
                                if c[2][1]>=70 and \
                                    not(not(False in (s in sym for s in c[2][0])) \
                                    or (not(False in (s in alpha+sym for s in c[2][0])) and len(c[2][0])<=2) \
                                    or len(c[2][0])<=1 or c[2][0]==len(c[2][0])*c[2][0][0]):

                                    print('text: ',c_3rd[2])
                                    inner_3rd= addDict("XCUIElementTypeButton","button",c_3rd[2][0],c_3rd[1])
                                else: 
                                    inner_3rd = addDict("XCUIElementTypeButton","button","",c[1])
                                    inner_3rd.pop("_EA@text", None)
                                    inner_3rd["_TaaD::imageToBeClipped"] = True
                                
                                for i_4th, c_4th in enumerate(second):
                                    
                                    if c_3rd[0][0]==c_4th[0][1][3]:
                                        print(c_4th[0][0],c_4th[0][1],c_4th[1],'4th')
                                        temp.append(c_4th[1])
                                        cnt+=1
                                        if c[2][1]>=70 and \
                                            not(not(False in (s in sym for s in c[2][0])) \
                                            or (not(False in (s in alpha+sym for s in c[2][0])) and len(c[2][0])<=2) \
                                            or len(c[2][0])<=1 or c[2][0]==len(c[2][0])*c[2][0][0]):

                                            print('text: ',c_4th[2])
                                            inner_4th = addDict("XCUIElementTypeButton","button",c_4th[2][0],c_4th[1])
                                        else: 
                                            inner_4th = addDict("XCUIElementTypeButton","button","",c[1])
                                            inner_4th.pop("_EA@text", None)
                                            inner_4th["_TaaD::imageToBeClipped"] = True
                                        
                                        inner_3rd["child"].append(inner_4th)
                                        
                                inner_2nd["child"].append(inner_3rd)
                        
                        inner_1st["child"].append(inner_2nd)
                secondLayer['child'].append(inner_1st)

        outerLayer['child'].append(secondLayer)
    
    print('cnt: ', cnt)
    
    with open(filename, 'w') as fp:
        json.dump(outerLayer, fp, indent=4)
    
    return outerLayer, temp

# "_EA@isEnabled": true, 
# "_EA@class": "XCUIElementTypeOther", 
# "_EA@isHidden": true,
# "_EA@isClickable": false, 
# "tag": "div", 
# "xPercentage": 37, 
# "yPercentage": 26, 
# "widthPercentage": 13, 
# "heightPercentage": 19, 
# "child": []
    
# "_EA@isEnabled": true,
# "_EA@class": ["XCUIElementTypeOther", "XCUIElementTypeImage", "XCUIElementTypeCell"], 
# "_EA@isHidden": false, 
# "_EA@isClickable": true, 
# "tag": ["div", "img", "button"], 
# "xPercentage": 37, 
# "yPercentage": 26, 
# "widthPercentage": 13, 
# "heightPercentage": 19, 
# "_EA@type": "button", 
# "child":[]
    
# "_EA@isEnabled": true, 
# "_EA@class": ["XCUIElementTypeOther", "XCUIElementTypeScrollView", "XCUIElementTypeWindow", "XCUIElementTypeApplication"], 
# "_EA@isHidden": false, 
# "_EA@isClickable": false, 
# "tag": ["div", "ul"], 
# "xPercentage": 0, 
# "yPercentage": 0, 
# "widthPercentage": 100, 
# "heightPercentage": 100, 
# "_EA@isScrollable": true, 
# "_EA@text": "Facebook", 
# "_EA@name": "Facebook", 
# "_FM@topic": "search_page", 
# "child":[]
    
#座標用絕對座標
#除了tag, 座標, child，前面都要加_EA@
#enabled->isEnabled
#"_EA@isClickable"
#type->class
#visible-> isHidden
#label -> text
#name -> name
#value -> value
#x->positionX->x
#y->positionY->y

#class, tag, child 是list
#label和value, name幾乎都是一樣的內容

# ##建立截圖的xml檔


# def addSubElement(elementType, label, parent, coordinate):
#     x,y,w,h = coordinate
#     a = ET.SubElement(parent, elementType, enabled="true", height=str(h), 
#                       label=label, type=elementType, visible="true", width=str(w), x=str(x), y=str(y))
#     return a

# def generateXML(filename, img, info):
    
#     h,w,_ = img.shape
#     appName = "app"
    
#     root = ET.Element('AppiumAUT')
#     app = ET.Element('XCUIElementTypeApplication', enabled="true", height=str(h), 
#                      label=appName, name=appName, type="XCUIElementTypeApplication", visible="true", width=str(w), x="0", y="0")
#     root.append(app)
#     window = ET.Element('XCUIElementTypeWindow', enabled="true", height=str(h), 
#                         type="XCUIElementTypeWindow", visible="true", width=str(w), x="0", y="0")
#     app.append(window)
#     other = ET.Element('XCUIElementTypeOther', enabled="true", height=str(h), 
#                        type="XCUIElementTypeOther", visible="true", width=str(w), x="0", y="0")
#     window.append(other)
    
    
#     cnt=0
#     for i, c in enumerate(info):
#         #同層下一個輪廓的序號、同層上一個輪廓的序號、子輪廓的序號、父輪廓的序號。
#         #[[輪廓編號, 輪廓hier資訊],[輪廓座標],輪廓文字]
#         #因為子輪廓有複數個的話hier裡只會有第一個的編號，所以要用子輪廓中hier的父輪廓編號來判斷
#         #所以不能用c[0][1][2]==c_2rd[0][0], 要用c[0][0]==c_2nd[0][1][3]
        
#         if c[0][1][3]==-1:
#             print(c[0][0],c[0][1])
#             cnt+=1
#             outer = addSubElement("XCUIElementTypeButton","outerLayer",other,c[1])
            
#             for i_2nd, c_2nd in enumerate(info):
                
#                 if c[0][0]==c_2nd[0][1][3]:
#                     print(c_2nd[0][0],c_2nd[0][1],'2nd')
#                     cnt+=1
#                     secondLayer = addSubElement("XCUIElementTypeButton","2ndLayer",outer,c_2nd[1])
                    
#                     for i_3rd, c_3rd in enumerate(info):
                
#                         if c_2nd[0][0]==c_3rd[0][1][3]:
#                             print(c_3rd[0][0],c_3rd[0][1],'3rd')
#                             cnt+=1
#                             thirdLayer = addSubElement("XCUIElementTypeButton","3rdLayer",secondLayer,c_3rd[1])
                            
#                             for i_4th, c_4th in enumerate(info):
                                
#                                 if c_3rd[0][0]==c_4th[0][1][3]:
#                                     cnt+=1
#                                     fourthLayer = addSubElement("XCUIElementTypeButton","4thLayer",thirdLayer,c_4th[1])
                
            
#     tree = ET.ElementTree(root)
    
    
#     tree.write(filename, encoding='utf-8', xml_declaration=True, pretty_print=True)
#     print(cnt)
    