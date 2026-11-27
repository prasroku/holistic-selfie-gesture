import cv2
import mediapipe as MP

from libs.coordinates_cv import all_fingers_up, calculate_coordinates, check_if_hand_is_open_and_straight
from libs.fps import FpsCounter

mp_hands = MP.solutions.hands
PHOTO_FILE = "photo.jpg"


def hand_coordinates(hand_landmarks, image) -> dict[int, list[float]]:
    """Pixel coordinates of one hand's 21 landmarks, keyed by landmark id."""
    h, w, _ = image.shape
    return {i: calculate_coordinates(lm, image, h, w) for i, lm in enumerate(hand_landmarks.landmark)}


def should_take_photo(results, image) -> bool:
    """True when any detected hand is an open palm with every finger pointing up."""
    for hand in results.multi_hand_landmarks or []:
        coords = hand_coordinates(hand, image)
        if check_if_hand_is_open_and_straight(coords) and all_fingers_up(coords):
            return True
    return False


def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands()
    fps = FpsCounter()
    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frame_fps = fps.tick()

            if should_take_photo(results, img):
                cv2.putText(
                    img,
                    f"{int(frame_fps)} FPS",
                    (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(PHOTO_FILE, img)

            cv2.imshow("Image", img)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
