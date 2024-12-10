import cv2
import mediapipe as mp
import numpy as np
import time
import random
import threading
from playsound import playsound
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2


# helper function for displaying png image
def overlay_image_alpha(background, overlay, x, y):
    """ Overlays a RGBA image on a BGR background at position (x, y) with alpha blending. """
    h, w = overlay.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return  # Out of bounds

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    bg_region = background[y:y+h, x:x+w]
    bg_region = (1.0 - mask) * bg_region + mask * overlay_img
    background[y:y+h, x:x+w] = bg_region.astype(np.uint8)

def play_sound_async(sound_file):
    """Play audio in a separate thread so it doesn't block the main thread."""
    def play():
        playsound(sound_file)
    threading.Thread(target=play, daemon=True).start()

#classes

class Cursor:
    def __init__(self, x, y, size=10):
        self.x = x
        self.y = y
        self.is_clicked = False
        self.is_dragging = False
        self.size = size
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.last_click_time = 0
        self.last_release_time = 0
        self.debounce_delay = 0.3
        self.click_state_changed = False
        self.click_event_handlers = []
        self.release_event_handlers = []

    def click(self):
        current_time = time.time()
        if current_time - self.last_click_time > self.debounce_delay and current_time - self.last_release_time > self.debounce_delay:
            if not self.is_clicked:
                self.is_clicked = True
                self.last_click_time = current_time
                self.click_state_changed = True
                self._trigger_click_events()

    def release(self):
        current_time = time.time()
        if current_time - self.last_release_time > self.debounce_delay and current_time - self.last_click_time > self.debounce_delay:
            if self.is_clicked:
                self.is_clicked = False
                self.is_dragging = False
                self.last_release_time = current_time
                self.click_state_changed = True
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
    def __init__(self, x, y, width, height, label, is_draggable=True, image=None):
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
        self.image = image

    def is_hovered(self, cursor):
        return (self.x - cursor.size <= cursor.x <= self.x + self.width + cursor.size and
                self.y - cursor.size <= cursor.y <= self.y + self.height + cursor.size)

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
        # if clicked than dragging follow the cursor
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

        # when button is FIRST clicked
        if clicked and not self.is_clicked_state:
            self.is_clicked_state = True
            self._trigger_click_first_events()
            self.dragging = True
            # capture offset so it stays at the sae orientation
            self.drag_offset_x = cursor.x - self.x
            self.drag_offset_y = cursor.y - self.y
        elif not clicked and self.is_clicked_state:
            # let go
            self.is_clicked_state = False
            self.dragging = False

        if clicked:
            self._trigger_click_events()

        if self.dragging and self.is_draggable:
            # button stays under cursor with offset
            self.x = cursor.x - self.drag_offset_x
            self.y = cursor.y - self.drag_offset_y

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

def draw_button(image, button):
    if button.image is not None and button.image.shape[2] == 4:
        scaled_img = cv2.resize(button.image, (button.width, button.height), interpolation=cv2.INTER_AREA)
        overlay_image_alpha(image, scaled_img, button.x, button.y)
        if button.is_hovered_state:
            cv2.rectangle(image, (button.x, button.y), (button.x + button.width, button.y + button.height), (0,255,0), 2)
    else:
        button_color = (0, 255, 0) if button.is_hovered_state else (0, 0, 255)
        if button.is_clicked_state:
            button_color = (0, 255, 255)
        cv2.rectangle(image, (button.x, button.y), (button.x + button.width, button.y + button.height), button_color, 2)
        cv2.putText(image, button.label, (button.x + 10, button.y + button.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, button_color, 2)

class Goose:
    def __init__(self, x, y, speed, goose_img):
        self.x = x
        self.y = y
        self.speed = speed
        self.goose_img = goose_img
        self.width = goose_img.shape[1]
        self.height = goose_img.shape[0]

    def update(self):
        self.y += self.speed

    def draw(self, image):
        overlay_image_alpha(image, self.goose_img, self.x, self.y)


# Main App
class GestureRecognizerApp:
    def __init__(self, model_path):
        self.cursor = Cursor(100, 100)
        self.last_detected_gesture_name = None
        self.last_detected_gesture_landmarks = None
        self.options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self.print_result,
            min_hand_detection_confidence=0.75
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.options)
        self.cursor.on_click(self.on_cursor_click)
        self.cursor.on_release(self.on_cursor_release)
        
        # open palm gesture needs to be held for 0.1s // prevent mis-inputs
        self.required_open_palm_time = 0.1
        self.open_palm_start = None

        # load images
        self.basket_img = cv2.imread("basket.png", cv2.IMREAD_UNCHANGED)
        self.goose_img = cv2.imread("goose.png", cv2.IMREAD_UNCHANGED)
        if self.basket_img is None:
            self.basket_img = np.zeros((80,80,4), dtype=np.uint8)
            self.basket_img[:,:,0:3] = 255
            self.basket_img[:,:,3] = 255
        else:
            self.basket_img = cv2.resize(self.basket_img, (80,80), interpolation=cv2.INTER_AREA)

        if self.goose_img is None:
            self.goose_img = np.zeros((50,50,4), dtype=np.uint8)
            self.goose_img[:,:,0:3] = 200
            self.goose_img[:,:,3] = 255
        else:
            self.goose_img = cv2.resize(self.goose_img, (50,50), interpolation=cv2.INTER_AREA)

        # Main screen buttons
        self.buttons = [
            Button(50, 50, 100, 50, "Button 1"),
            Button(200, 50, 100, 50, "Button 2"),
            Button(350, 50, 100, 50, "Button 3"),
            Button(500, 50, 100, 50, "Button 4"),
            Button(250, 400, 100, 50, "Reset", is_draggable=False),
            Button(400, 400, 150, 50, "Goose Grab", is_draggable=False)
        ]
        for button in self.buttons:
            button.on_hover(self.on_button_hover)
            button.on_hover_enter(self.on_button_hover_enter)
            button.on_hover_leave(self.on_button_hover_leave)
            button.on_click(self.on_button_click)
            button.on_click_first(self.on_button_click_first)

        self.buttons[-2].on_click(self.on_reset_button_click)
        self.buttons[-1].on_click(self.on_goose_grab_click)

        # mini-game variables
        self.in_minigame = False
        self.score = 0
        self.last_spawn_time = time.time()
        self.spawn_interval = 1.0
        self.gooses = []
        self.basket = Button(300, 300, 80, 80, "", image=self.basket_img.copy())
        self.basket.on_click(self.on_basket_click)
        self.basket.on_click_first(self.on_basket_click_first)

        self.back_button = Button(50,50,100,50,"Back", is_draggable=False)
        self.back_button.on_click(self.on_back_button_click)

    def print_result(self, result, output_image, timestamp_ms):
        if result.gestures:
            first_gesture = result.gestures[0][0]
            self.last_detected_gesture_name = first_gesture.category_name
        else:
            self.last_detected_gesture_name = None

        if result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in result.hand_landmarks[0]
            ])
            self.last_detected_gesture_landmarks = hand_landmarks_proto
        else:
            self.last_detected_gesture_landmarks = None

    def spawn_goose(self, frame_width):
        x = random.randint(0, frame_width - self.goose_img.shape[1])
        goose = Goose(x, 0, speed=5, goose_img=self.goose_img)
        self.gooses.append(goose)

    def update_gooses(self, image):
        to_remove = []
        for g in self.gooses:
            g.update()
            g.draw(image)
            # basket collision
            bx1, bx2 = self.basket.x, self.basket.x+self.basket.width
            by1, by2 = self.basket.y, self.basket.y+self.basket.height
            gx1, gx2 = g.x, g.x+g.width
            gy1, gy2 = g.y, g.y+g.height
            if not (gx2 < bx1 or gx1 > bx2 or gy2 < by1 or gy1 > by2):
                # caught
                self.score += 1
                play_sound_async("sound1.mp3")
                to_remove.append(g)
            elif g.y > image.shape[0]:
                # missed
                to_remove.append(g)
        for r in to_remove:
            self.gooses.remove(r)

    def handle_gestures_for_drag(self):
        # closed fist to hold and drag
        if self.last_detected_gesture_name == "Closed_Fist":
            self.open_palm_start = None
            if not self.cursor.is_clicked:
                self.cursor.click()
        # open palm to release
        elif self.last_detected_gesture_name == "Open_Palm":
            if self.cursor.is_clicked:
                if self.open_palm_start is None:
                    self.open_palm_start = time.time()
                else:
                    if time.time() - self.open_palm_start >= self.required_open_palm_time:
                        self.cursor.release()
        else:
            # other gestures do nothing
            # remains clicked if we were clicked before!!!
            self.open_palm_start = None

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            self.recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # move cursor based on landmarks
            if self.last_detected_gesture_landmarks:
                wrist = self.last_detected_gesture_landmarks.landmark[0]
                index_finger_mcp = self.last_detected_gesture_landmarks.landmark[5]
                palm_x = (wrist.x + index_finger_mcp.x) / 2
                palm_y = (wrist.y + index_finger_mcp.y) / 2
                self.cursor.move(int(palm_x * image.shape[1]), int(palm_y * image.shape[0]))

            # handle dragging logic based on gestures
            self.handle_gestures_for_drag()
            draw_cursor(image, self.cursor)

            # show recognized gesture on main screen
            if self.last_detected_gesture_name:
                cv2.putText(image, f"Gesture: {self.last_detected_gesture_name}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

            if not self.in_minigame:
                for button in self.buttons:
                    draw_button(image, button)
                    button.update_state(self.cursor)
            else:
                # Mini-game
                draw_button(image, self.basket)
                self.basket.update_state(self.cursor)

                draw_button(image, self.back_button)
                self.back_button.update_state(self.cursor)

                # spawns the gooses!
                if time.time() - self.last_spawn_time > self.spawn_interval:
                    self.last_spawn_time = time.time()
                    self.spawn_goose(image.shape[1])

                self.update_gooses(image)
                cv2.putText(image, f"Score: {self.score}", (image.shape[1]-150,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            cv2.imshow('MediaPipe Gesture Recognizer', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def on_cursor_click(self):
        pass

    def on_cursor_release(self):
        pass

    def on_button_hover(self):
        pass

    def on_button_hover_enter(self):
        pass

    def on_button_hover_leave(self):
        pass

    def on_button_click(self):
        pass

    def on_button_click_first(self):
        play_sound_async("sound4.mp3")

    def on_reset_button_click(self):
        for button in self.buttons:
            button.reset()

    def on_goose_grab_click(self):
        self.in_minigame = True
        self.score = 0
        self.gooses.clear()
        self.basket.x = 300
        self.basket.y = 300

    def on_basket_click(self):
        pass

    def on_basket_click_first(self):
        pass

    def on_back_button_click(self):
        self.in_minigame = False
        self.gooses.clear()



if __name__ == "__main__":
    model_path = Path(__file__).parent / "gesture_recognizer.task"
    app = GestureRecognizerApp(str(model_path))
    app.run()

