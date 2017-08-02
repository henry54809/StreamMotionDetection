import os
import datetime
import time
import detect
import numpy as np
import cv2
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
import smtplib
import math
import globals

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def draw_rects(img, rects, color, resize_ratio=1):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (math.floor(x1/resize_ratio), math.floor(y1/resize_ratio)), (math.floor(x2/resize_ratio), math.floor(y2/resize_ratio)), color, 2)

# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype="float") 
 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type

def pushbullet_notify(image_path):
    #push bullet notifications
    with open(image_path, 'rb') as image:
         image_data = globals.pb.upload_file(image, image_path)
    push = globals.pb.push_file(**image_data) 

def email_notify(image_paths, video_path, subject=""):
    max_images = int(globals.cfg['NOTIFICATION']['MaxAttachments']) - 1
    print('Sending notification emails')
    recipients = ['henry54809@gmail.com']
    msg = MIMEMultipart()
    msg['Subject'] = globals.cfg['NOTIFICATION']['SubjectPrefix'] + subject + globals.cfg['NOTIFICATION']['SubjectPostfix'] 
    msg['From'] = globals.cfg['NOTIFICATION']['Username'] 
    msg.preamble = 'Multipart massage.\n' 

    part = MIMEText(str(datetime.datetime.now()))
    msg.attach(part)
    print("Processing ", len(image_paths), " frames.")
    if len(image_paths) > max_images:
        image_paths = np.array(image_paths)
        index = np.random.randint(0, len(image_paths), max_images) 
        index = np.sort(index)
        image_paths = image_paths[index]
    for image_path in image_paths:
        with open(image_path, 'rb') as image:  
            part = MIMEApplication(image.read())
            part.add_header('Content-Disposition', 'attachment', filename=image_path)
            msg.attach(part)
    with open(video_path, 'rb') as video:  
        part = MIMEApplication(video.read())
        part.add_header('Content-Disposition', 'attachment', filename=video_path)
        msg.attach(part)
     
    session = smtplib.SMTP(globals.cfg['NOTIFICATION']['SMTPServer'], int(globals.cfg['NOTIFICATION']['SMTPPort']))
    session.ehlo()
    session.starttls()
    session.login(msg['From'], globals.cfg['NOTIFICATION']['Password'])
    session.sendmail(msg['From'], recipients, msg.as_string())
    session.quit()
    print('Notification emails sent')
