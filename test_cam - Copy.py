import cv2

cap = cv2.VideoCapture(0)  # try 0, 1, or 2
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured")
        break
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
