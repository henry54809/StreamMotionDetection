# StreamMotionDetection
Detects motion and faces in a RTSP/MJEG Stream. Program saves images and videos where motion is detected and emails the attachments to specified recipients.

## Dependencies
* OpenCV3 with Python3 bindings
* Python3
### Optional
* Pushbullet
* Pushbullet API key

## Overview
* Install Python3 and OpenCV3
* Run `import cv2` in a python3 shell to verify opencv3 installation was successful.
* Fill in configuration file (more info below)
* Start program with python app.py --config_file [YOUR CONFIG FILE]

## Sample Configuration File
```
[STREAM]
RTSPStream = 
Username = 
Password = 

[PROCESSING]
LiveFacialRecognition = False 
ResizeRatio = 0.2
FacialDetectionCascadeScale = 1.1
MotionDetectionMinArea = 100
Visual = True 
OutputPath = 

[NOTIFICATION]
#multivalue field; separate using ","
Recipients = 
SubjectPrefix = Motion Detected
SubjectPostfix = 
SMTPServer = 
SMTPPort = 
Username = 
Password = 
MaxAttachments = 11
MinRecordSeconds = 10
NotifyCoolDownSeconds = 60
PushbulletAPIKey = 
```
### Stream Configuration
* Fill in a url of either RTSPStream or MJPEGStream
* If applicable, fill in username and password of either stream

### Processing
* Fill in `OutputPath` where the program should save images.
#### Optional
* Modify `ResizeRatio` used in image processing if needed. Too high of a value may cause the program to seg fault.
* Modify `Visual` to change the visiblity of processing stream
* Modify `MotionDetectionMinArea` to ignore small areas of movement.
* Change `LiveFacialRecognition` to enable facial recognition(frontal and profile views). 
* Lower `FacialDetectionCascadeScale` to increase facial detection accuracy. A lower value will cause processing time to increase. 

### Notification
* Fill in `Recipients`. This is a multivalue field; separate emails by a comma.
* Fill in your email's `SMTPServer`, `SMTPPort`, `Username`, `Password`.
#### Optional
* Modify `MaxAttachments` if needed to keep attachments size under control.
* Modify `MinRecordSeconds` if needed to dictate when to stop recording after last dection of motion. This means, the program will continue to record if motion persists.
* Fill in `PushbulletAPIKey` to also send over motion images using pushbullet. Change `NotifyCoolDownSeconds` to contraint how often to send notifications using pushbullet.
