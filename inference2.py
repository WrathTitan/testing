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
    f = open("model_age.pkl", "rb")
    model = pickle.load(f)
    image = load(file_path)
    confidences = model.predict(image)

    actual = np.argmax(confidences)
    print(actual)
    if(actual == 0):
        return("Age: 0-25")
    elif(actual == 1):
        return("Age: 26-50")
    elif(actual == 2):
        return("Age: 51-75")
    else:
        return("Age: 76-100")


print(infer_image("1.jpg"))
print(infer_image("2.jpg"))


def infer_video():
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            f = open("model_age.pkl", "rb")
            model = pickle.load(f)

            np_image = np.array(np.asarray(frame)).astype('float32')/255
            np_image = transform.resize(np_image, (300, 300, 3))
            np_image = np.expand_dims(np_image, axis=0)

            confidences = model.predict(np_image)

            actual = np.argmax(confidences)
            print(actual)
            if(actual == 0):
                print("Age: 0-25")
                cv2.putText(frame, "Age: 0-25", (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
            elif(actual == 1):
                print("Age: 26-50")
                cv2.putText(frame, "Age: 26-50", (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
            elif(actual == 2):
                print("Age: 51-75")
                cv2.putText(frame, "Age: 51-75", (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
            else:
                print("Age: 76-100")
                cv2.putText(frame, "Age: 76-100", (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)

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
