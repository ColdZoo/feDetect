# -*-coding:utf-8-*-
import cv2
import numpy as np
import caffe
import os

SKIPCOUNT = 5  #跳帧数目

def fe_init():
    global caffe_root
    global net
    global transformer
    caffe.set_mode_gpu()
    net = caffe.Net('deploy.prototxt', 'fe_train_iter_40000.caffemodel', caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load('fe_test_mean.npy').mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].reshape(50, 3, 227, 227)

def fe_predict(file_name):
    global net
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(file_name))
    out = net.forward()
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    #print(labels[out['prob'].argmax()])
    index = out['prob'].argmax()
    #print(out['prob'][index, index])
    return [labels[out['prob'].argmax()], out['prob'][index, index]]


cv2.namedWindow('faceDetect')
cap = cv2.VideoCapture(0) #打开0号摄像头
success, frame = cap.read()#读取一桢图像，前一个返回值是是否成功，后一个返回值是图像本身
size = frame.shape[:2]#获得当前桢彩色图像的大小
color = (255, 0, 0)#设置人脸框的颜色
classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")#定义分类器
faceid = 0
fe_init()
skipcount = SKIPCOUNT#每SKIPCOUNT帧判断一帧
while success:
    success, frame = cap.read()
    image = np.zeros(size, dtype=np.float16)#定义一个与当前桢图像大小相同的的灰度图像矩阵
    image = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)#将当前桢图像转换成灰度图像
    cv2.equalizeHist(image, image)#灰度图像进行直方图等距化

    #如下三行是设定最小图像的大小
    divisor = 8
    h, w = size
    minSize = (w/divisor, h/divisor)
    faceRects = classifier.detectMultiScale(image, 1.1, 4, cv2.CASCADE_SCALE_IMAGE, minSize)#人脸检测
    if len(faceRects) > 0:  # 如果人脸数组长度大于0
        faceRect = faceRects[0]
        x, y, w, h = faceRect
        if w > 0 and skipcount == 0:
            skipcount = SKIPCOUNT
            im = image[y:y+w, x:x+h]
            save_name = 'test/face'+str(faceid)+'.jpg'
            faceid += 1
            im = cv2.resize(im, (227, 227))
            cv2.imwrite(save_name, im)
            [emotion, confidence] = fe_predict(save_name)
            os.remove(save_name) 
	    if confidence > 0.5:
                print(emotion)
        elif w > 0:
            skipcount -= 1
        cv2.rectangle(image, (x, y), (x+w, y+h), color)
    cv2.imshow("faceDetect", image)#显示图像

    key= cv2.waitKey(10)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break



cv2.destroyWindow("faceDetect")
