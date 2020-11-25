import cv2
import numpy as np
import constants

def preprocessing(frame, input_shape):
    # model format
    n, c, h, w = input_shape
    # resize frame to fit model spec
    dim = (h, w)
    image = np.copy(frame)
    image = cv2.resize(image, dim)
    # Rearrange image to CHW format
    image = image.transpose((2,0,1))
    # Reshape image to fit model
    image = image.reshape(n,c,h,w)

    return image

def get_face_coordinates(infer_output):
    result = infer_output[0][0]

    preds = [pred for pred in result if pred[1] == 1 and pred[2] > 0.8]
    coords = [[pred[3], pred[4], pred[5], pred[6]] for pred in preds]

    return coords

def main_face(coords):
    if coords:
        w_list = []
        for coord in coords:
            w_list.append(coord[2]) # get x_max
            main_face = np.where(coords == max(w_list))
            index = main_face[0][0] # get the first biggest face
            face = coords[index]
        return (face[0], face[1], face[2], face[3]) #return main face x,y,w,h

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

def draw_fps_time(frame, fps, face_infer, emo_infer, font, fontscale, color, thk):
    approx_fps = "Approx FPS : {:.2f}".format(fps.fps())
    face_time = "Face inference time : {:.6f}".format(face_infer)
    emo_time = "Emotion inference time : {:.6f}".format(emo_infer)

    cv2.putText(frame, approx_fps, (5, 20), font, fontscale, color, thk, cv2.LINE_AA, False)
    cv2.putText(frame, face_time, (5, 40), font, fontscale, color, thk, cv2.LINE_AA, False)
    cv2.putText(frame, emo_time, (5, 60), font, fontscale, color, thk, cv2.LINE_AA, False)

def draw_bounding_box(frame, coord, box_color, thickness):
    if coord:
        h, w, _ = frame.shape
        x1 = int(coord[0] * w)
        y1 = int(coord[1] * h)
        x2 = int(coord[2] * w)
        y2 = int(coord[3] * h)
        cv2.rectangle(frame, (x1,y1), (x2, y2), box_color, thickness)
        return (x1, y1, x2, y2)