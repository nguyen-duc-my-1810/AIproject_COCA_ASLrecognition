import cv2
import numpy as np

import json
import base64
import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket

import openai 
import mediapipe as mp

from keras.models import load_model
from textblob import TextBlob


# Load model and hand tracking
right_hand_model = load_model("../GUI/right_hand_model.h5")
left_hand_model = load_model("../GUI/left_hand_model.h5")
model = right_hand_model #Default

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def reversed_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    reversed_data = {v: k for k, v in data.items()}
    labels_dict = [reversed_data[i] for i in range(len(reversed_data))]
    return labels_dict

file_path = "../GUI/Index_to_letters.json"
labels_dict = reversed_json(file_path=file_path)

# Use OpenAI API
api_key = "sk-XXX" # Replace with your API key
openai.api_key = api_key

system_prompt = (
    "You are an English teacher. Students have presented incomplete sentences, and your task is to help them complete these sentences properly. Your goal is to provide grammatically correct and contextually fitting completions to the sentences given by the students. Ensure that the AI response is a complete sentence."
)
instruction_prompt = "Complete the following sentence: hi" 

def send_to_chatgpt(text, system_prompt=system_prompt, max_tokens=30, temperature=0.8):
    
    api_response = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    res = api_response.choices[0].message.content

    if res[-1] != '.':
        res = ".".join(res.split(sep=".")[:-1])
    return res



STABLE_FRAMES_THRESHOLD = 5
current_stable_frames = 0

stable_character = None
accumulated_word = ""
sentence1 = ""
sentence2 = ""
previous_character = ""

webcam_active = False 
stop_process = False

video_active = False
stop_video = False 

data = None
video_path = ""


app = FastAPI()
# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    return HTMLResponse(content=open("../GUI/COCA.html").read())

# Changing the model
@app.websocket("/model")
async def websocket_model(websocket: WebSocket):
    global model  

    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        received_data = json.loads(data)
        selected_model = received_data.get("selected_model")

        if selected_model:
            if selected_model == "right":
                model = right_hand_model
            elif selected_model == "left":
                model = left_hand_model
            else:
                pass
        print(f"Model updated: {selected_model}")
     
# Adjusting the recognition speed
@app.websocket("/threshold")
async def websocket_threshold(websocket: WebSocket):
    global STABLE_FRAMES_THRESHOLD

    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        received_data = json.loads(data)
        selected_threshold = received_data.get("selected_hand")
        if selected_threshold:
            STABLE_FRAMES_THRESHOLD = int(selected_threshold)
            print(f"STABLE_FRAMES_THRESHOLD updated: {STABLE_FRAMES_THRESHOLD}")

# Processing data and returning it to the frontend
async def process_webcam(websocket: WebSocket):
    global accumulated_word, sentence1, sentence2
    global STABLE_FRAMES_THRESHOLD
    global stop_process, webcam_active
    global data

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('videofps:', fps)
    
    stable_character = None
    accumulated_word = ""
    sentence1 = ""
    sentence2 = ""  

    while not stop_process:
        if data == '{"command":"deactivate_webcam"}':  
            stop_process = True
            webcam_active = False
        elif data == '{"command":"clear_accumulated_word"}':
            accumulated_word = ""
            sentence1 = ""
            sentence2 = ""

        data_aux = []
        x_ = []
        y_ = []
        z_ = []

        ret, frame = cap.read()
        if ret:
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            continue 

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            hand_landmarks = results.multi_hand_landmarks[0]

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                x_.append(x)
                y_.append(y)
                z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = z_[i]

                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                data_aux.append(z)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            input_data = np.array(data_aux)
            reshaped_data = input_data.reshape(1, 1, len(input_data))  

            prediction = model.predict(reshaped_data)
            predicted_index = np.argmax(prediction)
            predicted_character = labels_dict[predicted_index]

            if predicted_character == stable_character:
                current_stable_frames += 1
                stable_window = np.ones((500, 500, 3), dtype=np.uint8) * 255
                if current_stable_frames == STABLE_FRAMES_THRESHOLD:
                
                    if predicted_character == "space":
                        if previous_character == "space":
                            pass
                        else:
                            accumulated_word += " "
                            words_before_space = accumulated_word.split(" ")[:-1]
                            text_before_space = " ".join(words_before_space) 
                            print(text_before_space)

                            blob = TextBlob(text_before_space)
                            corrected_word = str(blob.correct())
                            print(corrected_word)
                            
                            if corrected_word != text_before_space:
                                print(f'Spelling Correction: {text_before_space} -> {corrected_word}')
                                accumulated_word = corrected_word
                                accumulated_word += " "

                            for i in range(2):
                                prompt = "Complete the sentence:" + accumulated_word
                                full_sentence = send_to_chatgpt(prompt)
                                if full_sentence is None or full_sentence.startswith("Complete the sentence:"):
                                    full_sentence = send_to_chatgpt(prompt)
                                if i == 0:
                                    result1 = full_sentence
                                else:
                                    result2 = full_sentence

                            sentence1 = result1
                            sentence2 = result2

                    elif predicted_character == "del":
                        accumulated_word = accumulated_word[:-1]    

                    elif predicted_character == "10":
                        accumulated_word = accumulated_word

                    elif predicted_character == "1":
                        if previous_character == "10":
                            accumulated_word = sentence1
                        else:
                            accumulated_word += predicted_character

                    elif predicted_character == "2":
                        if previous_character == "10": 
                            accumulated_word = sentence2
                        else:
                            accumulated_word += predicted_character
                    else:
                        accumulated_word += predicted_character
                    
                    accumulated_word = accumulated_word.lower() 

                    previous_character = predicted_character
                    print(f'Stable Character: {predicted_character}')
                    print(f'accumulated_word: {accumulated_word}')
        
                cv2.putText(stable_window, accumulated_word, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                data = {
                        "stable_window": base64.b64encode(cv2.imencode(".jpg", stable_window)[1]).decode("utf-8"),
                        "accumulated_word": accumulated_word,
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                    }
                await websocket.send_json(data)     
                await asyncio.sleep(0.1)
  
            else:
                stable_character = predicted_character
                current_stable_frames = 1

            predicted_character = predicted_character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = buffer.tobytes()
            
            await websocket.send_bytes(frame_base64)
            await asyncio.sleep(0.1)

# Activating the model upon pressing the camera button
@app.websocket("/ws")
async def websocket_webcam(websocket: WebSocket):
    global webcam_active
    global stop_process
    global data
    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        print(data)
        if data == '{"command":"activate_webcam"}':
            if not webcam_active:  
                webcam_active = True
                stop_process = False 
                await process_webcam(websocket)
            else:
                pass

        if data == '{"command":"deactivate_webcam"}':
            stop_process = False 
            await process_webcam(websocket)

# The video recognition function, but adjustments for handling an appropriate frames-per-second (fps) rate haven't been implemented yet.
async def process_video(websocket: WebSocket):
    global accumulated_word, sentence1, sentence2
    global STABLE_FRAMES_THRESHOLD
    global video_active, stop_video
    global data
    global video_path 

    stable_character = None
    accumulated_word = ""
    sentence1 = ""
    sentence2 = ""

    cap = cv2.VideoCapture(video_path)
 
    while not stop_video:
        if data == '{"command":"deactivate_video"}': 
            stop_video = True
            video_active = False
    
        data_aux = []
        x_ = []
        y_ = []
        z_ = []

        ret, frame = cap.read()
        if ret:
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            continue  

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            hand_landmarks = results.multi_hand_landmarks[0]

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                x_.append(x)
                y_.append(y)
                z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = z_[i]

                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                data_aux.append(z)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            input_data = np.array(data_aux)
            reshaped_data = input_data.reshape(1, 1, len(input_data))  

            prediction = model.predict(reshaped_data)
            predicted_index = np.argmax(prediction)
            predicted_character = labels_dict[predicted_index]

            if predicted_character == stable_character:
                current_stable_frames += 1
                stable_window = np.ones((500, 500, 3), dtype=np.uint8) * 255
                if current_stable_frames == STABLE_FRAMES_THRESHOLD:
                
                    if predicted_character == "space":
                        if previous_character == "space":
                            pass
                        else:
                            accumulated_word += " "
                            words_before_space = accumulated_word.split(" ")[:-1]
                            text_before_space = " ".join(words_before_space)  
                            print(text_before_space)

                    elif predicted_character == "L":
                        accumulated_word += "sl"

                    elif predicted_character == "D":
                        accumulated_word += "co"

                    elif predicted_character == "del":
                        accumulated_word = accumulated_word[:-1]    

                    else:
                        accumulated_word += predicted_character
                    
                    accumulated_word = accumulated_word.lower() 
                    previous_character = predicted_character
                    print(f'Stable Character: {predicted_character}')
                    print(f'accumulated_word: {accumulated_word}')
       
                cv2.putText(stable_window, accumulated_word, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                data = {
                        "stable_window": base64.b64encode(cv2.imencode(".jpg", stable_window)[1]).decode("utf-8"),
                        "accumulated_word": accumulated_word,
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                    }
                await websocket.send_json(data)  
                await asyncio.sleep(0.1)
  
            else:
                stable_character = predicted_character
                current_stable_frames = 1

            predicted_character = predicted_character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = buffer.tobytes()
            
            await websocket.send_bytes(frame_base64)
            await asyncio.sleep(0.1)

# Video selection function
# Because the "POST" method cannot be used, you can only replace the video file path on your device to "video_path"
@app.websocket("/ws_video")
async def websocket_video(websocket: WebSocket):
    global video_active, stop_video
    global data
    global video_path

    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        print(data)
        if data == '{"command":"activate_video"}':
            if not video_active:  
                video_active = True
                stop_video = False 
                # file_data = await websocket.receive_text()
                # file_data_dict = json.loads(file_data)
                video_path = "../GUI/video_test.mp4"

                await process_video(websocket)
            else:
                pass

        if data == '{"command":"deactivate_video"}':
            stop_video = False  
            await process_video(websocket)