from ultralytics import YOLO
import cv2
import os

model_name = 'yolo26n.pt'
folder_for_models = 'models'
path_to_model = os.path.join(folder_for_models, model_name) # Model will be saved in this app's folder

model = YOLO(path_to_model) 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow('YOLO', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






