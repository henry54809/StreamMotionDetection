import common
import hard_negative_svm.training as hard_negative 
import cv2
import numpy as np
import sys
import getopt
import threading
class FaceDetector:
     def __init__(self, scale=1.08):
         script_path = common.get_script_path()
         self.cascade = cv2.CascadeClassifier(script_path + "/haarcascade_frontalface_alt.xml")
         self.cascade_profile = cv2.CascadeClassifier(script_path + '/haarcascade_profileface.xml')
         self.scale = scale
         self.hog = cv2.HOGDescriptor()
         self.hog.load(script_path + '/hard_negative_svm/hog.xml')
         self.svm = cv2.ml.SVM_load(script_path + '/hard_negative_svm/output_frontal.xml')
         self.svm_profile = cv2.ml.SVM_load(script_path + '/hard_negative_svm/output_profile.xml')

     def detect(self, img):
         gray = self._process_img(img)
         for cascade in [self.cascade, self.cascade_profile]:
             faces = cascade.detectMultiScale(img, scaleFactor=self.scale, minNeighbors=2, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
             if len(faces) > 0:
                 faces[:,2:] += faces[:,:2]
                 common.non_max_suppression(faces, 0.5)
                 true_faces = []
                 not_faces = []
                 for rect in faces:
                     #hard negative svm predict
                     processed = hard_negative.process(img[rect[1]:rect[3], rect[0]:rect[2]])
                     processed_hog = np.transpose(self.hog.compute(processed))
                     results = self.svm.predict(processed_hog)
                     results_profile = self.svm_profile.predict(processed_hog)

                     if results[1].ravel()[0] == 1 or results_profile[1].ravel()[0] == 1:
                         true_faces.append(rect)
                     else:
                         not_faces.append(rect)
                 return (true_faces, not_faces)
         return ([], []) 

     def _process_img(self, img):
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
         gray = clahe.apply(gray)
         return gray

if __name__ == '__main__':
    #optlist, args = getopt.getopt(sys.argv[1:])
    face_detector = FaceDetector([1.05])
    image_path = sys.argv[1]
    img = cv2.imread(image_path, 1)
    (true_faces, not_faces) = face_detector.detect(img)
    print(true_faces, not_faces)
    if len(true_faces) > 0:
        common.draw_rects(img, true_faces, (0, 255, 0) )
        found = True
    if len(not_faces) > 0:
        common.draw_rects(img, not_faces, (225, 0, 0))
    cv2.imshow('window', img)
    while(True):
        cv2.waitKey(1)
