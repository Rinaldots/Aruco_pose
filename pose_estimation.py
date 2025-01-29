import cv2
import numpy as np

def pose_estimation(frame, aruco_dict, camera_matrix, dist_matrix):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if len(corners) > 0:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)

        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_matrix, rvec[i], tvec[i], 0.1)

    return frame

camera = cv2.VideoCapture(0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Define camera matrix and distortion coefficients
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    dist_matrix = np.zeros((5, 1))

    frame = pose_estimation(frame, aruco_dict, camera_matrix, dist_matrix)
    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()