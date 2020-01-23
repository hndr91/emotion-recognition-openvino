from openvino.inference_engine import IECore, IENetwork
import cv2
import numpy as np

def preprocessing_image(frame, frameSize=64):
    # model format
    n, c, h, w = [1, 3, frameSize, frameSize]
    # resize frame
    dim = (frameSize, frameSize)
    image = np.copy(frame)
    image = cv2.resize(image, dim)
    # Change order to CHW
    image = image.transpose((2,0,1))
    # Reshape image to fit model
    image = image.reshape(n,c,h,w)
    
    return image


cascade_model = cv2.CascadeClassifier("/home/rm6-polinema/intel/openvino_2019.3.376/opencv/etc/haarcascades/haarcascade_frontalface_default.xml")
model_xml = 'model/emotions-recognition-retail-0003.xml'
model_bin = 'model/emotions-recognition-retail-0003.bin'
DEVICE = "MYRIAD"

plugin = IECore()
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
exec_net = plugin.load_network(net, DEVICE)

# Load Image
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.5
color = (255,0,0)
thk = 1

while True:
    _,img = cap.read()
    height, width = img.shape[:2]
    org_y = 15
    
    #convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect face
    faces = cascade_model.detectMultiScale(gray, 1.1, 4)
    # try to get the 1st face only
    #face = faces[0]
    for (x,y,w,h) in faces:
        
        img_face = img[y:y+h,x:x+w]
    	
        prep_img = preprocessing_image(img_face)
    
        res = exec_net.infer(inputs={input_blob:prep_img})
        res = res["prob_emotion"]
        neutral = "Neutral {:.0%}".format(res[0][0][0][0])
        happy = "Happy {:.0%}".format(res[0][1][0][0])
        sad = "Sad {:.0%}".format(res[0][2][0][0])
        surprise = "Surprise {:.0%}".format(res[0][3][0][0])
        anger = "Anger {:.0%}".format(res[0][4][0][0])
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        
        cv2.putText(img, neutral, (width-100, org_y), font, fontscale, color, thk, cv2.LINE_AA, False)
        cv2.putText(img, happy, (width-100, org_y*2), font, fontscale, color, thk, cv2.LINE_AA, False)
        cv2.putText(img, sad, (width-100, org_y*3), font, fontscale, color, thk, cv2.LINE_AA, False)
        cv2.putText(img, surprise, (width-100, org_y*4), font, fontscale, color, thk, cv2.LINE_AA, False)
        cv2.putText(img, anger, (width-100, org_y*5), font, fontscale, color, thk, cv2.LINE_AA, False)
    
    cv2.putText(img, plugin.get_metric(DEVICE, "FULL_DEVICE_NAME"), (10, org_y), font, fontscale, color, thk, cv2.LINE_AA, False)
    cv2.imshow('capture', img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
