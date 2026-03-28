from gpiozero import Button
from signal import pause

# GPIO17 = physical pin 11
button = Button(17, pull_up=True, bounce_time=0.1)

def say_hello():
    print("hello world")

button.when_pressed = say_hello

print("Ready. Press the button...")

pause()
