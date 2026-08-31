import cv2, glob
demo = cv2.imread(sorted(glob.glob("demos/pick3/ep000/cam1_*.jpg"))[0])
c = cv2.VideoCapture(2, cv2.CAP_DSHOW)
c.set(cv2.CAP_PROP_FRAME_WIDTH, 640); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ok, live = c.read(); c.release()
cv2.imshow("demo (recorded)", demo)
cv2.imshow("live (now)", live)
cv2.waitKey(0)