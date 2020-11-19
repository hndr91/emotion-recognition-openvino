import argparse
import cv2
import numpy as np
from utils import preprocessing
from inference import Network
import os

# CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    f_desc = "The location of face recogntion model XML file"
    # e_desc = "The location of emotion recognition model XML file"
    d_desc = "Selected device. Default device is CPU"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-f", help=f_desc, required=True)
    # required.add_argument("-e", help=e_desc, required=True)
    optional.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()

    return args

def infer_on_video(args):
    
    # Initialize the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(args.f, args.d)
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
        # _, width = frame.shape[:2]

        p_frame = preprocessing(frame, net_input_shape)
        print(p_frame)

        # Perform async inference on the frame
        plugin.async_inference(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()

            print(result)



        # if face:
        #     (x, y, w, h) = face

        #     # get face in color based on face detection
        #     face = frame[y:y+h, x:x+w]
        #     p_frame = preprocessing(face)

        #     # Perform async inference on the frame
        #     plugin.async_inference(p_frame)

        #     # Get the output of inference
        #     if plugin.wait() == 0:
        #         result = plugin.extract_output()
            
        #     # result = result["prop_emotion"]
        #     # print(result)

        #     #draw box on faces
        #     draw_bounding_box(frame, x, y, w, h, constants.BOX_COLOR, constants.BOX_THICKNESS)

        #     # Draw emotion to frames
        #     draw_emotion(frame, result, constants.FONT, constants.FONT_SCALE, constants.FONT_COLOR, constants.THICKNESS, width)
        
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

