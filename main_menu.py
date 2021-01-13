import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

keyboard = np.zeros((600, 500, 3), np.uint8)
keys_set = {0: "Keyboard", 1: "Mouse",2:"Constructs"}
board = np.zeros((300, 1400), np.uint8)
board[:] = 255


def draw_letters(letter_index, text, letter_light):
        # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 0
        y = 200
    elif letter_index == 2:
        x = 0
        y = 400
    

    width = 500
    height = 200
    th = 3 # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 5
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
        
    else:

        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)
        


def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))


    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)


    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

# Counters
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 6
frames_active_letter = 9
text=""

while True:
    _, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
    rows, cols, _ = frame.shape
    keyboard[:] = (26, 26, 26)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Draw a white space for loading bar
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)

    #if select_keyboard_menu is True:
    #    draw_menu()
    # Keyboard selected
    active_letter = keys_set[letter_index]
    # Face detection
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye, right_eye = eyes_contour_points(landmarks)
        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Eyes color
        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)
        if blinking_ratio > 5:
            blinking_frames += 1
            frames -= 1              
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)               
            if blinking_frames == frames_to_blink:
                if active_letter != "":
                    text += active_letter
                if text=="Mouse":
                    cap.release()
                    cv2.destroyWindow('frame')
                    cv2.destroyWindow('MainMenu')
                    import mouse
                    cap = cv2.VideoCapture(0)
                    keyboard = np.zeros((600, 1000, 3), np.uint8)
                    continue
                elif text=="Keyboard":
                    cap.release()
                    cv2.destroyWindow('frame')
                    cv2.destroyWindow('MainMenu')
                    import keyboard
                    cap = cv2.VideoCapture(0)
                    keyboard = np.zeros((600, 1000, 3), np.uint8)
                    continue
                elif text=="Constructs":
                    cap.release()
                    cv2.destroyWindow('frame')
                    cv2.destroyWindow('MainMenu')
                    import constructs
                    cap = cv2.VideoCapture(0)
                    keyboard = np.zeros((600, 1000, 3), np.uint8)
                    continue
        else:
            blinking_frames = 0    
        # Display letters on the keyboard
        if frames == frames_active_letter:
        
            letter_index += 1
            frames = 0
            
        if letter_index == 3:
            letter_index = 0
        for i in range(3):
            text=""
            if i == letter_index:
                light = True
            else:
                light = False
            draw_letters(i, keys_set[i], light)
    # Blinking loading bar
        percentage_blinking = blinking_frames / frames_to_blink
        loading_x = int(cols * percentage_blinking)
        cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
        cv2.imshow("Frame", frame)
        cv2.imshow("MainMenu",keyboard)
        
        key = cv2.waitKey(1)
        if key == 27:
            break


cap.release()
cv2.destroyAllWindows()

        
