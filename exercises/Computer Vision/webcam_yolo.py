import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, conf=0.5)
    annotated_frame = results[0].plot()
    cv.imshow("YOLOv8 Live", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()