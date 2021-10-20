'''
PyPower Projects
Detect Gender and Age using Artificial Intelligence

'''

#Usage 
# Step 1 : Go to command prompt and set working directory where the gender_age.py file is stored
# Step 2 : Execute the following command to detect from image: python gender_age.py -i 1.jpg  
# Step 3 : Execute the following command to detect from webcam: python gender_age.py


# Import required modules
import os
import cv2 as cv
import cv2 as cv2
import math
import time
import argparse
import numpy as np


path = r'D:\python-project\Project8-Gender_Age_AI - Copy\Project8-Gender_Age_AI\images'
imagelist = os.listdir(path)


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 1)
    return frameOpencvDnn, bboxes



#parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
#parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

#args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNetFromCaffe(ageProto,ageModel)
genderNet = cv.dnn.readNetFromCaffe(genderProto,genderModel)
faceNet = cv.dnn.readNet(faceModel,faceProto)

def do_mosaic(frame, x, y, w, h, neighbor=20): #neighbor控制大小
      fh, fw = frame.shape[0], frame.shape[1]
      if (y + h > fh) or (x + w > fw):
            return
      for i in range(0, h - neighbor, neighbor): # 关键点0 减去neightbour 防止溢出
            for j in range(0, w - neighbor, neighbor):
                  rect = [j + x, i + y, neighbor, neighbor]
                  color = frame[i + y][j + x].tolist() # 关键点1 tolist
                  left_up = (rect[0], rect[1])
                  right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1) # 关键点2 减去一个像素
                  cv2.rectangle(frame, left_up, right_down, color, -1)

def recog(frame):    
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        #continue

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        print("Gender : {}, confidence = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        print("Age : {}, confidence = {:.3f}".format(age, agePreds[0].max()))

        label = "{}".format(gender) #gender,age
        if gender == 'Female':
            do_mosaic(frameFace,bbox[0]-5, bbox[1]-10,bbox[2]-bbox[0],bbox[3]-bbox[1])
            print('yes she is a female!')
        cv.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv.LINE_AA)
        #frameFace = frameFace[bbox[1]-40:bbox[1]+bbox[3]-20,bbox[0]-40:bbox[0]+bbox[2]-20] #crop the head
        #cv.imshow("Age Gender Demo", frameFace)
        bg = cv2.imread('bg.jpg')
        #frameFace = cv2.resize(frameFace,(0,0),fx=0.3,fy=0.3)  #缩放头部大小
        #print(frameFace.shape[:2])
        frameFace = cv2.resize(frameFace,(154,154),interpolation=cv2.INTER_CUBIC)
        bg = cv2.resize(bg,(0,0),fx=0.5,fy=0.5) #缩放背景
        
        rows, cols = frameFace.shape[:2] #获取head的高度、宽度
        roi = bg[250:rows+250, 160:cols+160]  #从第几行到第几行，第几列到第几列设为roi区域
        
       
        #frameFace = 255 - frameFace
        dst = cv2.addWeighted(frameFace,1,roi,0,0) #图像融合
        add_img = bg.copy() #对原图像进行拷贝
        add_img[250:rows+250, 160:cols+160] = dst  # 将融合后的区域放进原图
        #invert = 255 - add_img #反色
        mono = cv2.cvtColor(add_img, cv2.COLOR_BGR2GRAY) #单色
        cv.imshow("Age Gender Demo", mono)
        #name = args.i
        #cv.imwrite('./detected/'+name,frameFace)
    print("Time : {:.3f}".format(time.time() - t))



# Open a video file or an image file or a camera stream
padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    #frame = cv2.imread('./images/2.jpg')
    #cv2.waitKeSy(0)

    for imgname in imagelist:
 
        if(imgname.endswith(".jpg")):
 
            frame1 = cv2.imread(os.path.join(path,imgname))
            recog(frame1)
 
            #cv2.imshow("picture",image)

 
             # 每张图片的停留时间
            k = cv2.waitKey(3000)
 
            # 通过esc键终止程序
            if k == 27:
 
                break
 
    cv2.destroyAllWindows()
    

