import numpy as np
import cv2
import matplotlib.pylab as plt

#masking out the region of interest
def region( img, coordinates):
    mask = np.zeros_like(img)
    matchMaskColor = 255
    cv2.fillPoly(mask, coordinates, matchMaskColor)
    maskedImage = cv2.bitwise_and(img, mask)
    return maskedImage

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2),(0,255,0), thickness=3)

    #merging images
    img= cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    # to define the region in which we want (as a tuple)
    region_coordinates = [ (0, height), (width/2, height/2), (width, height)]
    
    #applying edge detection and hough line transform
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(grayscale_image, 100, 200)
    cropped_image = region(image, np.array([region_coordinates], np.int32))
    lines = cv2.HoughLinesP(canny_image, 6, np.pi/60, threshold = 160,minLineLength = 40,maxLineGap = 25)
    image_with_lines = draw_the_lines(image, lines)
    return (image_with_lines)

cap = cv2.VideoCapture('sample_output.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frames = process(frame)
    cv2.imshow('frame', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     
cap.release()
cv2.destroyAllWindows()