import threading
import time
import socket
import json

import cv2 as cv
import numpy as np
import pyaudio
from vosk import Model as VoskModel, KaldiRecognizer
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

#Smart Car setup
HOST = #Insert Elegoo Smart Car IP (it's usually 192.168.4.1)
PORT = #Insert the port (we did 100)

commands = {
    "forward": b'{"H":0,"N":3,"D1":3,"D2":150}',
    "backward": b'{"H":0,"N":3,"D1":4,"D2":150}',
    "left": b'{"H":0,"N":3,"D1":1,"D2":100}',
    "right": b'{"H":0,"N":3,"D1":2,"D2":100}',
    "spin": b'{"H":0,"N":3,"D1":1,"D2":250}',
}

def send_command(command_bytes, delay=1.5): #This is what sends the commands to the car
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b"{Heartbeat}") #"Heartbeat" is something with the car we have yet to understand but it is needed
            time.sleep(0.3)
            s.sendall(command_bytes)
            print(f"Sent: {command_bytes}")
            time.sleep(delay)
            s.sendall(commands["stop_movement"])
            print("Movement stopped")
    except Exception as e:
        print(f"Could not send command: {e}")

#Voice Recognition Thread
def voice_control(running_event):
    print("Voice control started. Say 'forward', 'left', etc. Say 'stop' to quit.")

    model = VoskModel("vosk-model-small-en-us-0.15") #This is the voice to text model we used
    recognizer = KaldiRecognizer(model, 16000)
    mic = pyaudio.PyAudio() #Where we get the audio from, it will grab from your default microphone
    stream = mic.open(format=pyaudio.paInt16, channels=1,
                      rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    while running_event.is_set():
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            time.sleep(0.05)
            result = json.loads(recognizer.Result())
            spoken = result["text"].lower()
            print(f"You said: {spoken}")


            if "camera" in spoken: #Activates the camera
                print("Turning on camera!")
                time.sleep(2)
                camera_thread = threading.Thread(target=camera_detection, args=(running,))
                camera_thread.start()
                camera_thread.join()

                
            if "stop" in spoken: #Stops the program
                print("Stop command received. Shutting down.")
                break
                

            matched = False
            for cmd in commands:
                if cmd in spoken and cmd != "stop_movement":
                    send_command(commands[cmd], delay=1)
                    matched = True
                    break

            if not matched and spoken != "":
                print("Command not recognized. Say 'forward', 'left', etc.")

    stream.stop_stream()
    stream.close()
    mic.terminate()

#Camera and Detection Thread
class CustomDepthwiseConv2D(DepthwiseConv2D): #This helps detect the object, helps the camera know what it is looking at
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

def camera_detection(running_event):
    print("Camera stream and object detection started.")

    model = load_model("Image Model/keras_model.h5", compile=False, #The image model that is also provided on this page
                       custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D})
    class_names = open("Image Model/labels.txt", "r").readlines()

    cam = cv.VideoCapture('http://192.168.4.1:81/stream')

    while running_event.is_set():
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        data = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)
        data = np.asarray(data, dtype=np.float32).reshape(1, 224, 224, 3)
        data = (data / 127.5) - 1

        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

#Prints out the class and confidence score

        if confidence_score >= 0.90:
            detected_obj = class_name[2:].strip()
            if detected_obj in ["Shoes", "Feet", "Face"]:
                print(f"Detected {detected_obj}! Confidence: {confidence_score:.2f}")
                send_command(commands["spin"], delay=2)

        cv.imshow('Camera', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Closing camera.")
            cam.release()
            cv.destroyAllWindows()

#Main
if __name__ == "__main__":
    running = threading.Event()
    running.set()

    voice_thread = threading.Thread(target=voice_control, args=(running,))
    voice_thread.start()
    voice_thread.join()

    print("Program ended.")

    voice_thread.join()
