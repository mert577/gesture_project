import mediapipe as mp
import cv2
import numpy as np
import os
import time
from playsound import playsound

from mediapipe.framework.formats import landmark_pb2

class Cursor:
    def __init__(self, x, y,size=10):
        self.x = x
        self.y = y
        self.is_clicked = False
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.last_click_time = 0
        self.last_release_time = 0
        self.debounce_delay = 0.3
        self.click_state_changed = False
        self.click_event_handlers = []
        self.release_event_handlers = []
        self.size = size

    def click(self):
        current_time = time.time()
        if current_time - self.last_click_time > self.debounce_delay and \
           current_time - self.last_release_time > self.debounce_delay:
            if not self.is_clicked:
                self.is_clicked = True
                self.last_click_time = current_time
                self.click_state_changed = True
                print("Click registered")
                self._trigger_click_events()

    def release(self):
        current_time = time.time()
        if current_time - self.last_release_time > self.debounce_delay and \
           current_time - self.last_click_time > self.debounce_delay:
            if self.is_clicked:
                self.is_clicked = False
                self.is_dragging = False
                self.last_release_time = current_time
                self.click_state_changed = True
                print("Release registered")
                self._trigger_release_events()

    def drag(self, x, y):
        if self.is_clicked:
            self.is_dragging = True
            self.drag_start_x = self.x
            self.drag_start_y = self.y
            self.x = x
            self.y = y

    def move(self, x, y):
        self.x = x
        self.y = y

    def on_click(self, handler):
        self.click_event_handlers.append(handler)

    def on_release(self, handler):
        self.release_event_handlers.append(handler)

    def _trigger_click_events(self):
        for handler in self.click_event_handlers:
            handler()

    def _trigger_release_events(self):
        for handler in self.release_event_handlers:
            handler()

def draw_cursor(image, cursor):
    if cursor.is_clicked:
        cv2.circle(image, (cursor.x, cursor.y), cursor.size+10, (0, 255, 0), -1)
    elif cursor.is_dragging:
        cv2.line(image, (cursor.drag_start_x, cursor.drag_start_y), (cursor.x, cursor.y), (0, 255, 0), 2)
    else:
        cv2.circle(image, (cursor.x, cursor.y), cursor.size, (0, 255, 0), -1)

class Button:
    def __init__(self, x, y, width, height, label, is_draggable=True):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.original_x = x
        self.original_y = y
        self.hover_event_handlers = []
        self.hover_enter_event_handlers = []
        self.hover_leave_event_handlers = []
        self.click_event_handlers = []
        self.click_first_event_handlers = []
        self.is_hovered_state = False
        self.is_clicked_state = False
        self.dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.is_draggable = is_draggable
        self.sound_file = None

    def is_hovered(self, cursor):
        return self.x - cursor.size <= cursor.x <= self.x + self.width + cursor.size and \
               self.y - cursor.size <= cursor.y <= self.y + self.height + cursor.size

    def is_clicked(self, cursor):
        return self.is_hovered(cursor) and cursor.is_clicked

    def on_hover(self, handler):
        self.hover_event_handlers.append(handler)

    def on_hover_enter(self, handler):
        self.hover_enter_event_handlers.append(handler)

    def on_hover_leave(self, handler):
        self.hover_leave_event_handlers.append(handler)

    def on_click(self, handler):
        self.click_event_handlers.append(handler)

    def on_click_first(self, handler):
        self.click_first_event_handlers.append(handler)

    def _trigger_hover_events(self):
        for handler in self.hover_event_handlers:
            handler()

    def _trigger_hover_enter_events(self):
        for handler in self.hover_enter_event_handlers:
            handler()

    def _trigger_hover_leave_events(self):
        for handler in self.hover_leave_event_handlers:
            handler()

    def _trigger_click_events(self):
        for handler in self.click_event_handlers:
            handler()

    def _trigger_click_first_events(self):
        for handler in self.click_first_event_handlers:
            handler()

    def update_state(self, cursor):
        hovered = self.is_hovered(cursor)
        clicked = self.is_clicked(cursor)

        if hovered and not self.is_hovered_state:
            self.is_hovered_state = True
            self._trigger_hover_enter_events()
        elif not hovered and self.is_hovered_state:
            self.is_hovered_state = False
            self._trigger_hover_leave_events()

        if hovered:
            self._trigger_hover_events()

        if clicked and not self.is_clicked_state:
            self.is_clicked_state = True
            self._trigger_click_first_events()
            self.dragging = True
            self.drag_offset_x = cursor.x - self.x
            self.drag_offset_y = cursor.y - self.y
        elif not clicked and self.is_clicked_state:
            self.is_clicked_state = False
            self.dragging = False

        if clicked:
            self._trigger_click_events()

        if self.dragging and self.is_draggable:
            self.x = cursor.x - self.drag_offset_x
            self.y = cursor.y - self.drag_offset_y

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

def draw_button(image, button):
    button_color = (0, 255, 0) if button.is_hovered_state else (0, 0, 255)
    if button.is_clicked_state:
        button_color = (0, 255, 255)
    cv2.rectangle(image, (button.x, button.y), (button.x + button.width, button.y + button.height), button_color, 2)
    cv2.putText(image, button.label, (button.x + 10, button.y + button.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, button_color, 2)

class GestureRecognizerApp:
    def __init__(self, model_path):
        self.cursor = Cursor(100, 100)
        self.last_detected_gesture_name = None
        self.last_detected_gesture_landmarks = None
        self.options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.print_result,
            min_hand_detection_confidence=0.75  # Set the minimum detection confidence to 0.75
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.options)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Register event handlers
        self.cursor.on_click(self.on_cursor_click)
        self.cursor.on_release(self.on_cursor_release)

        # Create buttons
        self.buttons = [
            Button(50, 50, 100, 50, "Button 1"),
            Button(200, 50, 100, 50, "Button 2"),
            Button(350, 50, 100, 50, "Button 3"),
            Button(500, 50, 100, 50, "Button 4"),
            Button(250, 400, 100, 50, "Reset", is_draggable=False)
        ]

        # Register button events
        for button in self.buttons:
            button.on_hover(self.on_button_hover)
            button.on_hover_enter(self.on_button_hover_enter)
            button.on_hover_leave(self.on_button_hover_leave)
            button.on_click(self.on_button_click)
            button.on_click_first(self.on_button_click_first)

        # Register reset button event
        self.buttons[-1].on_click(self.on_reset_button_click)

    def print_result(self, result, output_image, timestamp_ms):
        if result.gestures:
            first_gesture = result.gestures[0][0]
            self.last_detected_gesture_name = first_gesture.category_name
        if result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in result.hand_landmarks[0]
            ])
            self.last_detected_gesture_landmarks = hand_landmarks_proto

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            if not success:
                print("Ignoring empty camera frame.")
                continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            self.recognizer.recognize_async(mp_image, frame_timestamp_ms)

            if self.last_detected_gesture_name:
                cv2.putText(image, self.last_detected_gesture_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            draw_cursor(image, self.cursor)

            if self.last_detected_gesture_landmarks:
   
                wrist = self.last_detected_gesture_landmarks.landmark[0]
                index_finger_mcp = self.last_detected_gesture_landmarks.landmark[5]
                
                #palm is between wrist and index finger mcp
                palm_x = (wrist.x + index_finger_mcp.x) / 2
                palm_y = (wrist.y + index_finger_mcp.y) / 2
                
                
                self.cursor.move(int(palm_x * image.shape[1]), int(palm_y * image.shape[0]))

            if self.last_detected_gesture_name == 'Closed_Fist':
                self.cursor.click()
            else:
                self.cursor.release()

            for button in self.buttons:
                draw_button(image, button)
                button.update_state(self.cursor)

            cv2.imshow('MediaPipe Gesture Recognizer', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def on_cursor_click(self):
        print("Cursor clicked event triggered")

    def on_cursor_release(self):
        print("Cursor released event triggered")

    def on_button_hover(self):
        print("Button hover event triggered")

    def on_button_hover_enter(self):
        print("Button hover enter event triggered")

    def on_button_hover_leave(self):
        print("Button hover leave event triggered")

    def on_button_click(self):
        print("Button click event triggered")

    def on_button_click_first(self):
        print("Button click first event triggered")
        playsound("sound4.mp3", False)

    def on_reset_button_click(self):
        print("Reset button clicked")
        for button in self.buttons:
            button.reset()

if __name__ == "__main__":
    model_path = r'C:\Users\Mert POLAT\Dersler\CS 449\gesture_project\gesture_recognizer.task'
    app = GestureRecognizerApp(model_path)
    app.run()