
# coding: utf-8

# In[15]:

import urllib.request
import numpy as np
import sys
import os
import re
import cv2
import hard_negative_svm.training as hard_negative 
import math
import threading
from multiprocessing import Process
import datetime
import time
import queue
import globals

def draw_rects(img, rects, color, resize_ratio=1):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (math.floor(x1/resize_ratio), math.floor(y1/resize_ratio)), (math.floor(x2/resize_ratio), math.floor(y2/resize_ratio)), color, 2)

def draw_str(dst, pos, s):
    x,y = pos
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()


bytes = bytes()
q = queue.Queue()
def get_rtsp_stream_image(stream):
    while True:
        assert(stream.isOpened())
        ret, frame = stream.read()
        assert(ret)
        q.put_nowait(frame) 
        cv2.waitKey(1)

def get_mjpeg_stream_image(stream):
    global bytes
    while True:
        bytes += stream.read(20000)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a!= -1 and b != -1:
             jpg = bytes[a:b+2]
             bytes = bytes[b+2:]
             img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
             q.put_nowait(img)
        cv2.waitKey(1)
 
def process_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    return gray 

if __name__ == '__main__':
    print("started at: ", str(datetime.datetime.now()))
    globals.init(sys.argv[1:])
    cfg = globals.cfg 
    import detect 
    import common
     
    if not cfg:
        sys.exit('config is not defined.')
    
    output_path = cfg['PROCESSING']['OutputPath']

    print("Detected images are stored under " + output_path )
    if cfg['PROCESSING'].getboolean('LiveFacialRecognition'):
        print("Using live face detection")
        face_detector = detect.FaceDetector(float(cfg['PROCESSING']['FacialDetectionCascadeScale'])) 

    if cfg['STREAM'].get('MJPEGstream'):
        # create an authorization handler
        p = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        if cfg['STREAM']['Username'] and cfg['STREAM']['Password']:
            p.add_password(None, cfg['STREAM']['MJPEGstream'], cfg['STREAM']['Username'], cfg['STREAM']['Password']);

        auth_handler = urllib.request.HTTPBasicAuthHandler(p)

        opener = urllib.request.build_opener(auth_handler)

        urllib.request.install_opener(opener)
        try:
            stream = opener.open(cfg['STREAM']['MJPEGstream'])
        except IOError as e:
            print (e)
            sys.exit('Could not open mjpeg stream')

        stream_thread = threading.Thread(target=get_mjpeg_stream_image, args=(stream,))
        stream_thread.start()
    elif cfg['STREAM'].get('RTSPStream'):
        if cfg['STREAM']['Username'] and cfg['STREAM']['Password']:
            stream = cv2.VideoCapture(cfg['STREAM']['Username'] + ':' + cfg['STREAM']['Password'] + '@' + cfg['STREAM']['RTSPStream'])
        else:
            stream = cv2.VideoCapture(cfg['STREAM']['RTSPStream'])
        assert(stream.isOpened())
        
        stream_thread = threading.Thread(target=get_rtsp_stream_image, args=(stream,))
        stream_thread.start()
    else:
        sys.exit('Stream is not defined')
            
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    resize_ratio = float(cfg['PROCESSING']['ResizeRatio'])
    notify_cooldown_seconds = int(cfg['NOTIFICATION']['NotifyCoolDownSeconds'])
    min_record_seconds = int(cfg['NOTIFICATION']['MinRecordSeconds'])
    min_area = float(cfg['PROCESSING']['MotionDetectionMinArea'])
    last_notify = clock() 
    last_found = None 
    t = clock()
    avg = None
    recorded_frames = []
    recording = False

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5,5))
    print("Waiting for stream to initialize...")
    img = q.get(timeout=5)
    print("Done")

    while True:
         if not stream_thread.is_alive():
             sys.exit("Dead stream thread.")
         pt = clock()
         img = q.get(timeout=5)
         q.task_done()
         img_copy = img.copy()
         height, width = img.shape[:2]
         i = cv2.resize(img_copy, (math.floor(width * resize_ratio), math.floor(height * resize_ratio)))
         
         gray = process_img(i)
         found = False

         #Motion detection
         if avg is None:
             avg = gray.copy().astype("float")
             continue
         
         cv2.accumulateWeighted(gray, avg, 0.5)
         frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
         thresh = cv2.threshold(frameDelta, 8, 255,
             cv2.THRESH_BINARY)[1]
         kernel = np.ones((5,5),np.uint8)
         thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

         _, cnts, __ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         to_draw = [] 
         for c in cnts:
             if cv2.contourArea(c) < min_area:
                 continue
             to_draw.append(c)

         if len(to_draw) > 0:
             c = to_draw[0]
             (x, y, w, h) = cv2.boundingRect(c)
             draw_rects(img_copy, [(x, y, x+w, y+h)], (255, 255, 0), resize_ratio=resize_ratio)
             found = True

         #Face detection
         if (found or recording) and cfg['PROCESSING'].getboolean('LiveFacialRecognition'):
             (true_faces, not_faces) = face_detector.detect(i)
             if len(true_faces) > 0:
                 draw_rects(img_copy, true_faces, (0, 255, 0), resize_ratio=resize_ratio)
                 found = True
             if len(not_faces) > 0:
                 draw_rects(img_copy, not_faces, (225, 0, 0), resize_ratio=resize_ratio)

         if recording:
             video.write(img_copy)

         if found:
             if last_found is None:
                 last_found = clock()
             if not recording:
                 video_path = output_path + '/v_' + str(clock()) + '.mov'
                 video = cv2.VideoWriter(video_path, fourcc, 25.0, (width, height))
                 video.write(img_copy)
                 recording = True
             image_path = output_path + '/' + str(clock()) + '.jpg'
             cv2.imwrite(image_path, img_copy)     
             #Pushbullet Notification of the first frame
             if hasattr(globals, 'pb') and (clock() - last_notify) > notify_cooldown_seconds:
                 last_notify = clock()
                 Process(target=common.pushbullet_notify, args=(image_path,)).start()
             
             #Purge queue at this instant
             for _ in range(q.qsize()):
                video.write(q.get())
                q.task_done()   

             if(clock() - last_found) < min_record_seconds: 
                recorded_frames.append(image_path)
                last_found = clock() 
             else:
                video.release()
                process = Process(target=common.email_notify, args=(recorded_frames, video_path))
                process.start()
                recorded_frames = []
                recording = False
                last_found = None
         #Flush image sequences
         elif recording and (clock() - last_found) >= min_record_seconds: 
             video.release()
             process = Process(target=common.email_notify, args=(recorded_frames, video_path))
             process.start()
             recorded_frames = []
             recording = False
             last_found = None 
         dt = clock() - t
         if cfg['PROCESSING'].getboolean('Visual'):
             draw_str(img_copy, (20, 20), '%.1f fps' % (1/dt))
             draw_str(img_copy, (20, 35), 'process: %.1f ms' % ((clock() - pt) * 1000))
             cv2.imshow('stream', img_copy)
         else:
             pass

         cv2.waitKey(1)
         t = clock()
    cv2.destroyAllWindows()
