import cv2

#Enabling Camera since it is Local Camera we use 0
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 240)

# Cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#layer for detecting eyes
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
#layer for detecting Smile
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Face
#Corners around Face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)

#drawing bounding box around face

    for (x, y, w, h) in faces:
     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#Eye
    eyes = eyeCascade.detectMultiScale(imgGray)
# drawing bounding box for eyes
    for (ex, ey, ew, eh) in eyes:
     img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 3)
#Smile
    smile = smileCascade.detectMultiScale(imgGray)
# drawing boundings box for smile 
    for (zx, zy, zw, zh) in smile :
       img = cv2.rectangle(img, (zx, zw), (zx+zw, zy+zh), (255,0,0),3)
    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')