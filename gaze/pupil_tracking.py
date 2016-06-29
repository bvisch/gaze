import cv2
import numpy as np
from math import floor
from PupilFinder import PupilFinder

def find_face(frame, face_cascade):
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces[0]


def find_eyes(face):

    left_c = [int(floor(0.3 * face.shape[0])), int(floor(0.4 * face.shape[1]))]
    right_c = [int(floor(0.7 * face.shape[0])), int(floor(0.4 * face.shape[1]))]

    size = int(floor(0.07 * face.shape[0]))
    left_x, left_y = [left_c[0]-size, left_c[1]-size]
    right_x, right_y = [right_c[0]-size, right_c[1]-size]
    area = size*2

    left_eye = (left_x, left_y, area, area)
    right_eye = (right_x, right_y, area, area)

    return [left_eye, right_eye]


def find_pupil(eye):
    # smooth eye to dull specular lighting
    eye = cv2.GaussianBlur(eye,(5,5),0.01*eye.shape[1])

    # calculate all distance vectors
    eye_len = np.arange(eye.shape[0])
    xx,yy = np.meshgrid(eye_len,eye_len)
    X1,X2 = np.meshgrid(xx.ravel(),xx.ravel()) # [1,2,3,1,2,3,1,2,3] & [1,1,1,1,1,1,1,1,1] ([2,2,2,2,2,2,2,2,2] would be next)
    Y1,Y2 = np.meshgrid(yy.ravel(),yy.ravel()) # [1,1,1,2,2,2,3,3,3] & [1,1,1,1,1,1,1,1,1] ([1,1,1,1,1,1,1,1,1] would be next)
    Dx,Dy = [X1-X2,Y1-Y2]
    Dlen = np.sqrt(Dx**2+Dy**2)
    Dx,Dy = [Dx/Dlen,Dy/Dlen] #normalized

    # get gradient
    Gx,Gy = np.gradient(eye.astype('float32'))
    Gmagn = np.sqrt(Gx**2+Gy**2)
    Gx,Gy = [Gx/Gmagn,Gy/Gmagn] #normalized
    GX,GY = np.meshgrid(Gx.ravel(),Gy.ravel())

    X = (GX*Dx+GY*Dy)
    #X[X<0] = 0
    X = X**2

    # weight darker areas higher by multiplying by inverted image
    eye = cv2.bitwise_not(eye)
    eyem = np.repeat(eye.ravel()[np.newaxis,:],eye.size,0)
    C = (np.nansum(eyem*X, axis=0)/eye.size).reshape(eye.shape)

    mask = np.ones(C.shape, dtype=bool)
    mask[1:-1,1:-1] = False
    C[mask] = 0
    #threshold = 0.9*C.max()
    #C[C<threshold] = 0
    #retval,C = cv2.threshold(C.astype(np.float32), threshold, 0, cv2.THRESH_TOZERO)

    return np.unravel_index(C.argmax(), C.shape)[::-1]



#
# MAIN LOOP
#

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
pupil_finder = PupilFinder()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    try:
        (x,y,w,h) = find_face(gray, face_cascade)

        # Draw box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = find_eyes(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eye_gray = roi_gray[ey:ey+eh,ex:ex+ew]
            eye_color = roi_color[ey:ey+eh,ex:ex+ew]
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            px,py = pupil_finder.find_pupil(eye_gray)
            cv2.rectangle(eye_color,(px,py),(px+1,py+1),(255,0,255),2)
    except:
        pass

    #Display the resulting frame
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
