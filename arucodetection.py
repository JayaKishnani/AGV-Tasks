import numpy as np
import cv2
import sys, time, math

id_to_find  = 23
marker_size  = 10 

#camera calibrating factors
Camera_matrix= np.loadtxt('camera_matrix.txt')
Distortion_coefficients = np.loadtxt('camera_distortion.txt')

#180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

#defining aruco tag
dictionary  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters  = cv2.aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 7                                        #default value

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])   #returns cos(0y)

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])                    #x= tan(-1)= R[2,1]/R[2,2]
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

#normal size sans-serif font
font = cv2.FONT_HERSHEY_SIMPLEX 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

#setting height and width of frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    corners, ids, rejected = cv2.aruco.detectMarkers(image=grayscale, dictionary=dictionary, parameters=parameters,
                              cameraMatrix=Camera_matrix, distCoeff=Distortion_coefficients)
    
    #check of ids list isn't empty 
    if ids is not None and ids[0] == id_to_find:
        
        ret = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, Camera_matrix, Distortion_coefficients)
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

        # detect contour around tag
        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.aruco.drawAxis(frame, Camera_matrix, Distortion_coefficients, rvec, tvec, 10)

        positiont = "Marker position  x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, positiont, (0,100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        #Rotation matrix from camera to tag
        R_ct    = np.matrix(cv2.Rodrigues(rvec)[0])
        #Rotation matrix from tag to camera
        R_tc    = R_ct.T

        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_tc)

        rotation_parametersm = "Marker rotation parameters roll=%4.0f  pitch=%4.0f  yaw=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),
                            math.degrees(yaw_marker))
        cv2.putText(frame, rotation_parametersm, (0,150), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        pos_camera = -R_tc*np.matrix(tvec).T

        positionc = "Camera Position x=%4.0f  y=%4.0f  z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2])
        cv2.putText(frame, positionc, (0, 200), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip*R_tc)
        rotation_parametersc = "Camera rotation parameters roll=%4.0f  pitch=%4.0f  yaw=%4.0f"%(math.degrees(roll_camera),math.degrees(pitch_camera),
                            math.degrees(yaw_camera))
        cv2.putText(frame, rotation_parametersc, (0, 250), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    cv2.imshow('ARUCO DETECTION FRAME', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break






