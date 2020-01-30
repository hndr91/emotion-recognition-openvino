import argparse
import cv2
import numpy as np
from inference import Network
import constants

# CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    d_desc = "The device name, if not 'MYRIAD'"
    c_desc = "The location of The OpenCV Cascade Haar model"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-c", help=c_desc, required=True)
    optional.add_argument("-d", help=d_desc, default='MYRIAD')
    args = parser.parse_args()

    return args

def detect_faces(frame, cascade_model):
    #convert frame to gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # get faces
    faces = cascade_model.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=4, minSize=(200,200))

    if type(faces) is not tuple:
        w = []
        for i in range(faces.shape[0]):
            w.append(faces[i][2])
            main_face = np.where(faces == max(w))
            index = main_face[0][0]
            face = faces[index]
            return (face[0], face[1], face[2], face[3]) #return main face x,y,w,h

def preprocessing(frame, frame_size=64):
    # model format
    n, c, h, w = [1, 3, frame_size, frame_size]
    # resize frame to fit model spec
    dim = (frame_size, frame_size)
    image = np.copy(frame)
    image = cv2.resize(image, dim)
    # Rearrange image to CHW format
    image = image.transpose((2,0,1))
    # Reshape image to fit model
    image = image.reshape(n,c,h,w)

    return image

def draw_emotion(frame, result, font, font_scale, font_color, thickness, width):
    org_y = 15

    # Get emotion from inference result
    neutral = "Neutral {:.0%}".format(result[0][0][0][0])
    happy = "Happy {:.0%}".format(result[0][1][0][0])
    sad = "Sad {:.0%}".format(result[0][2][0][0])
    surprise = "Surprise {:.0%}".format(result[0][3][0][0])
    anger = "Anger {:.0%}".format(result[0][4][0][0])

    # Draw emotion on frame
    cv2.putText(frame, neutral, (width-100, org_y), font, font_scale, font_color, thickness, cv2.LINE_AA, False)
    cv2.putText(frame, happy, (width-100, org_y*2), font, font_scale, font_color, thickness, cv2.LINE_AA, False)
    cv2.putText(frame, sad, (width-100, org_y*3), font, font_scale, font_color, thickness, cv2.LINE_AA, False)
    cv2.putText(frame, surprise, (width-100, org_y*4), font, font_scale, font_color, thickness, cv2.LINE_AA, False)
    cv2.putText(frame, anger, (width-100, org_y*5), font, font_scale, font_color, thickness, cv2.LINE_AA, False)

def draw_bounding_box(frame, x, y, h, w, box_color, thickness):
    cv2.rectangle(frame, (x,y), (x+w, y+h), box_color, thickness)

def infer_on_video(args):
    # Load Haar Cascade Model
    cascade_model = cv2.CascadeClassifier(args.c)

    # Initialize the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(args.m, args.d)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture from cam
    cap = cv2.VideoCapture(0)
    
    # Process frames until process is exited
    while True:
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Get frame's width
        _, width = frame.shape[:2]

        # Detect Main Face Only
        face = detect_faces(frame, cascade_model)
        if face:
            (x, y, w, h) = face

            # get face in color based on face detection
            face = frame[y:y+h, x:x+w]
            p_frame = preprocessing(face)

            # Perform async inference on the frame
            plugin.async_inference(p_frame)

            # Get the output of inference
            if plugin.wait() == 0:
                result = plugin.extract_output()
            
            # result = result["prop_emotion"]
            # print(result)

            #draw box on faces
            draw_bounding_box(frame, x, y, w, h, constants.BOX_COLOR, constants.BOX_THICKNESS)

            # Draw emotion to frames
            draw_emotion(frame, result, constants.FONT, constants.FONT_SCALE, constants.FONT_COLOR, constants.THICKNESS, width)
        
        #Show Frame
        cv2.imshow('Emotion Recognition', frame)

        # Break if escape key pressed
        key_pressed = cv2.waitKey(30) & 0xff
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()

