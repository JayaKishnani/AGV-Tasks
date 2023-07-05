import cv2
import numpy as np
import glob
import os

# define checkerboard size
checkerboard = (6, 9)

# termination when accuracy and max. iterations are done
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# array to store 3D and 2D point vectors
objpoints = []
imgpoints = []

objp = np.zeros((6*9, 3), np.float32)
# creates multidimenstional mesh grid
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# finding files recursively
files = glob.iglob('*.jpg', recursive=True)

for file in files:
    image = cv2.imread(file)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(grayscale, (6, 9), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            grayscale, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # to draw corners
        img = cv2.drawChessboardCorners(image, checkerboard, corners2, ret)

    cv2.imshow('ImageFinal', img)
    cv2.imwrite('Output.jpg', img)
    

ret, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grayscale.shape[::-1],None,None)

print(" Camera matrix:")
print(matrix)
print("\n Distortion coefficients:")
print(distortion)
print("\n Rotation Vectors:")
print(rvecs)
print("\n Translation Vectors:")
print(tvecs)

Output = cv2.FileStorage('Data.yaml', cv2.FILE_STORAGE_WRITE)
Output.write("camera_matrix", matrix)
Output.write("distortion_coefficients", distortion)

Output.release()
cv2.waitKey(0)

