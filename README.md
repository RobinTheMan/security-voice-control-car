# security-voice-control-car
This is a very very simple project that uses a AI Image Reading model and a Voice Model to identify people and take voice controls from the user
The idea is that it is almost like a security car, once it detects a person it will spin to try and "alert" the owner of the car, however due to some limitations with the car, it can only really send a message like that

# This project was done in collaboration with
[WilliamC1234](https://github.com/WilliamC1234) and [bbeltra](https://github.com/bbeltra)

# Hardware Components
You are gonna need one of these cars
[Elegoo Smart Car v4](https://us.elegoo.com/products/elegoo-smart-robot-car-kit-v-4-0?srsltid=AfmBOoqPuPuBWAyRUrXPEGc86NutonWYCew_cnta1a6-40fRDaGKVDom)

This is what the project is based entirely around, you may be able to edit it to function with another car, but I am not sure how that will go

# Instructions
Here are the things you need to make this program function:

- [Opencv](https://pypi.org/project/opencv-python/)
- [Numpy](https://numpy.org/install/)
- [Pyaudio](https://pypi.org/project/PyAudio/)
- [A voskmodel](https://alphacephei.com/vosk/models) For this one it may be best to go for the 1.8gb one since it's the accurate generic one
- [Tensorflow model](https://www.tensorflow.org/) This is for the camera, however I will put the model that was used for this project here so it is easier to get started

Once you have all of those, the program should technically work

# Connecting to the car
Now a lot of this is on the car and the program, you need to be able to turn on your car and connect to it
You need to add the address of the car to the program and the port into the program
[Car manual](https://drive.google.com/drive/folders/1FmqoM8KrJYJkFXHQnQBTL6xnNQ4cgTCa)
This link should hopefully help guide you into the specifics of your car and how to connect to it

Once you connect to it this could should run if you just execute it with little changes needed to the Arduino IDE

The program should run fine and after a bit of adjusting it should be able to pick up your voice and you can start giving it commands

