
from pynput.keyboard import Key, Listener
import threading
"""
Class that obtains the human feedback from the computer's keyboard.

Usage: 
1. first selection an axis you want to change, like, x, y or z
2. selection the direction along the axis by the arrows on the keyboard.
"""


class Feedback_keyboard_1d:
    def __init__(self, key_type, h_up, h_down, h_right, h_left, h_null):
        self.h = [0]
        self.h_null = [0]
        self.h_up = str_2_array(h_up)
        self.h_down = str_2_array(h_down)
        self.h_right = 1
        self.h_left = -1
        self.h_direction = None
        self.restart = False
        self.evaluation = False
        self.model_training = False

        # Starting the listener
        listener_thread = threading.Thread(target=self.start_listener)
        listener_thread.start()
    
    def start_listener(self):
        # This function will run the listener in the background
        with Listener(on_press=self.key_press, on_release=self.key_release) as listener:
            listener.join()

    def key_press(self, k):
        if k == Key.left:
            self.h = [-1]
        elif k == Key.right:
            self.h = [1]
        # elif k == Key.up:
        #     self.h = [0, 1]
        # elif k== Key.down:
        #     self.h = [0, -1]
        elif k == Key.up:  # switch up and down for push T env
            self.h = [1]
        elif k== Key.down:
            self.h = [-1]
        elif k == Key.space:
            self.restart = True
  
    def key_release(self, k):
        if k in (Key.left, Key.right, Key.up, Key.down):
            print("key release")
            # self.h = self.h_null

    def get_h(self):
        received_h = self.h
        self.h = self.h_null
        return received_h

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done
