import tensorflow as tf
import os
import cv2
import pickle

from PIL import Image
import numpy as np
from skimage import transform

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (300, 300, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


def infer_image(file_path):
    f = open("model.pkl", "rb")
    model = pickle.load(f)
    image = load(file_path)
    confidence = model.predict(image)

    if(confidence > 0.7):
        return "Male"
    else:
        return "Female"


print(infer_image("1.jpg"))
print(infer_image("2.jpg"))


def infer_video():
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            f = open("model.pkl", "rb")
            model = pickle.load(f)

            np_image = np.array(np.asarray(frame)).astype('float32')/255
            np_image = transform.resize(np_image, (300, 300, 3))
            np_image = np.expand_dims(np_image, axis=0)

            confidence = model.predict(np_image)

            if(confidence > 0.7):
                print("Male")
                cv2.putText(frame, "Male", (0,0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
            else:
                print("Female")
                cv2.putText(frame, "Female", (0,0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)

            cv2.imshow("Video Feed",frame)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break
        
    except KeyboardInterrupt as e:
        print("Error: ",e)
        print("Keyboard Interrupt encountered... Terminating Video Stream...")

    except Exception as e:
        print("An Error Occured: ",e)

    finally:
        cap.release()
        cv2.destroyAllWindows()


infer_video()
