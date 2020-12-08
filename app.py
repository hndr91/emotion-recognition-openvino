import argparse
import cv2
import numpy as np
from utils import preprocessing, get_face_coordinates, main_face, draw_bounding_box, draw_emotion, draw_fps_time, connect_mqtt, get_emotion
from inference import Network
import constants
from imutils.video import FPS
import time

import sys
import logging as log
import json

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    f_desc = "The location of face recogntion model XML file"
    e_desc = "The location of emotion recognition model XML file"
    d_desc = "Selected device. Default device is CPU"
    l_desc = "CPU extension path"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-f", help=f_desc, required=True)
    required.add_argument("-e", help=e_desc, required=True)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-l", help=l_desc, default=constants.CPU_EXTENSION)
    args = parser.parse_args()

    return args

def infer_on_video(args, client):
    
    # Initialize the Inference Engine
    face_detect_plugin = Network()
    emot_detect_plugin = Network()

    # Load the network model into the IE
    ## Face model
    face_detect_plugin.load_model(args.f, args.d, args.l)
    face_net_input_shape = face_detect_plugin.get_input_shape()
    ## Emotion model
    emot_detect_plugin.load_model(args.e, args.d)
    emot_net_input_shape = emot_detect_plugin.get_input_shape()

    # Get and open video capture from cam
    cap = cv2.VideoCapture(0)
    
    # Process frames until process is exited
    while True:
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break

        _, frame_width,_ = frame.shape
        face_frame = preprocessing(frame, face_net_input_shape)

        fps = FPS().start()
        # Perform async inference on the frame
        face_start_time = time.time()
        face_detect_plugin.async_inference(face_frame)
        face_end_time = time.time() - face_start_time

        # Get the output of inference
        if face_detect_plugin.wait() == 0:
            face_detect_result = face_detect_plugin.extract_output()
        
        coords = get_face_coordinates(face_detect_result)
        face_coord = main_face(coords)
        face = draw_bounding_box(frame, face_coord, constants.BOX_COLOR, constants.BOX_THICKNESS)

        emot_end_time = 0
        if face:
            (x, y, w, h) = face

            # get face in color based on face detection
            face_region = frame[y:y+h, x:x+w]
            if face_region.size:
                emot_frame = preprocessing(face_region, emot_net_input_shape)

            # Perform async inference on the frame
            emot_start_time = time.time()
            emot_detect_plugin.async_inference(emot_frame)
            emot_end_time = time.time() - emot_start_time
            

            # Get the output of inference
            if emot_detect_plugin.wait() == 0:
                emot_result = emot_detect_plugin.extract_output()

            # Get emotion prediction
            neutral, happy, sad, surprise, anger = get_emotion(emot_result)
            
            # Publish to MQTT Server
            client.publish(constants.MQTT_TOPIC1, json.dumps({
                "neutral" : neutral,
                "happy" : happy,
                "sad" : sad,
                "surprise" : surprise,
                "anger" : anger
            }))

            # Draw emotion to frames
            draw_emotion(frame, emot_result, constants.FONT, constants.FONT_SCALE, constants.FONT_COLOR, constants.THICKNESS, frame_width)
        

        fps.update()
        fps.stop()
        
        draw_fps_time(frame, fps, face_end_time, emot_end_time, constants.FONT, constants.FONT_SCALE, constants.FONT_COLOR, constants.THICKNESS)

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
    client = connect_mqtt()
    infer_on_video(args, client)


if __name__ == "__main__":
    main()

