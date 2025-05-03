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

#Robot Connection Setup
HOST = ""#Insert IP
PORT = ""#Insert port number (not as a string)

#These are your commands
commands = {
    "forward": b'{"H":0,"N":3,"D1":3,"D2":150}',
    "backward": b'{"H":0,"N":3,"D1":4,"D2":150}',
    "left": b'{"H":0,"N":3,"D1":1,"D2":100}',
    "right": b'{"H":0,"N":3,"D1":2,"D2":100}',
    "spin": b'{"H":0,"N":3,"D1":1,"D2":250}',
    "stop_movement": b'{"H":0,"N":100}',
}

last_action_time = 0
cooldown_seconds = 10
action_in_progress = threading.Event()

#This is how the commands are sent
def send_command(command_bytes, delay=1.5):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b"{Heartbeat}") #The heartbeat can be removed but when this was programmed it was difficult to remove so it's part of this one
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

    #Uses the vosk model for recognizing voice
    model = VoskModel("vosk-model-small-en-us-0.15")
    recognizer = KaldiRecognizer(model, 16000)
    mic = pyaudio.PyAudio()
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

            matched = False

            if "forward" in spoken:
                send_command(commands["forward"])
                matched = True
            elif "backward" in spoken:
                send_command(commands["backward"])
                matched = True
            elif "left" in spoken:
                send_command(commands["left"])
                matched = True
            elif "right" in spoken:
                send_command(commands["right"])
                matched = True
            elif "spin" in spoken:
                send_command(commands["spin"])
                matched = True
            elif "camera" in spoken:
                print("Turning on camera!")
                time.sleep(2)
                camera_thread = threading.Thread(target=camera_detection, args=(running_event,))
                camera_thread.start()
                matched = True
                print("Press 'q' to exit Camera Mode!" )
            elif "exit" in spoken:
                print("Stop command received. Shutting down.")
                send_command(commands["stop_movement"], delay=0.5)
                running_event.clear()
                matched = True

            if not matched and spoken != "":
                print("Command not recognized. Say 'forward', 'left', etc.")

    stream.stop_stream()
    stream.close()
    mic.terminate()

#Camera and Detection Thread
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

def perform_spin_action():
    try:
        send_command(commands["spin"])
        print("Spinning...")
        time.sleep(5)
        send_command(commands["stop_movement"])
        print("Done spinning. Resuming detection.")
    finally:
        action_in_progress.clear()

def camera_detection(running_event):
    print("Camera stream and object detection started.")

    model = load_model("Image Model/keras_model.h5", compile=False,
                       custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D})
    class_names = open("Image Model/labels.txt", "r").readlines()

    cam = cv.VideoCapture('http://192.168.4.1:81/stream')
    if not cam.isOpened():
        print("[ERROR] Cannot open video stream")
        return

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

        if confidence_score >= 0.90 and not action_in_progress.is_set():
            detected_obj = class_name[2:].strip()
            if detected_obj in ["Shoes", "Feet", "Face"]:
                print(f"Detected {detected_obj}!")
                action_in_progress.set()
                threading.Thread(target=perform_spin_action).start()

        cv.imshow('Camera', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Closing camera.")
            running_event.clear()
            break

    cam.release()
    cv.destroyAllWindows()

# ------------------ Main Program ------------------
if __name__ == "__main__":
    running = threading.Event()
    running.set()

    voice_thread = threading.Thread(target=voice_control, args=(running,))
    voice_thread.start()
    voice_thread.join()