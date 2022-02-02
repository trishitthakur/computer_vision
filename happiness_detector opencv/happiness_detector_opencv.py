'''from computer vision A-Z course
   coded by trishit nath thakur'''

#code implemented in virtual env

# Importing the libraries


import cv2


# Loading the cascades


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') # We load the cascade for the smile.


# Defining a function that will do the detections


def detect(gray, frame):

    # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.

           # scaling factor 1.3 means size of image will be reduced 1.3 times and 5 is number of neighbor zones that need to be accepted

    for (x, y, w, h) in faces:
         
           # For each detected face: x and y cordinates of upper left corner, w is width and h is height for rectangles

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

          # We paint a rectangle around the face. input is upper left and lower right cordinates then, color and thickness of edges of rectangle

        roi_gray = gray[y:y+h, x:x+w]  # We get the region of interest in the black and white image.

        roi_color = frame[y:y+h, x:x+w]  # We get the region of interest in the colored image.

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22) # We apply the detectMultiScale method to locate one or several eyes in the image.

          # scaling factor 1.1 means size of image will be reduced 1.1 times and 3 is number of neighbor zones that need to be accepted

        for (ex, ey, ew, eh) in eyes:  # For each detected eye: ex and ey cordinates of upper left corner, ew is width and eh is height for rectangles

            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22) # similar to above

        for (sx, sy, sw, sh) in smiles:

            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)


    return frame  # We return the image with the detector rectangles.


# Doing some Face Recognition with the webcam


video_capture = cv2.VideoCapture(0) # We turn the webcam on. 0 because we have inbuilt camera

while True:  # We repeat infinitely (until break):

    _, frame = video_capture.read()  # We get the last frame.
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations to convert image to black and white

    canvas = detect(gray, frame)   # We get the output of our detect function.

    cv2.imshow('Video', canvas)  # We display the outputs.

    if cv2.waitKey(1) & 0xFF == ord('q'):  # If we type on the keyboard:q the camera will be closed

        break  # We stop the loop.

video_capture.release() # We turn the webcam off.

cv2.destroyAllWindows()  # We destroy all the windows inside which the images were displayed.