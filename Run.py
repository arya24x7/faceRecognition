from keras.models import load_model
import cv2
import numpy as np
from random import choice


REV_CLASS_MAP = {
    0: "Prajwal",
    1: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]





model = load_model("faceRecognitionModel.h5")


cap = cv2.VideoCapture(0)

prev_move = None
computer_move_name = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    
            

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your name: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    if user_move_name == "Prajwal":
        print(user_move_name)
    else:
        print("??????")
        
    cv2.imshow("Face Recognition", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
