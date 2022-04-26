import cv2

url = 'mp/test4.html'

cap = cv2.VideoCapture(url)
while True:
    ret, img = cap.read()
    cv2.imshow('stream', img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()