import cv2
import numpy as np
from queue import PriorityQueue
import math
import time

path="Task_1_Low.png"
img=cv2.imread(path)
cv2.namedWindow("map",cv2.WINDOW_NORMAL)

obstacle_color=(255,255,255)
start_color=(45,204,113)
end_color=(231,76,60)

m,n,l=img.shape

for x in range(m):
    for y in range(n):
        b,g,r=img[x,y]
        if b==start_color[2] and g==start_color[1] and r==start_color[0]:
            source=(x,y)
        if b==end_color[2] and g==end_color[1] and r==end_color[0]:
            end=(x,y)

dist = np.full((m, n), np.inf)

q = PriorityQueue()

class Node:
    def __init__(self,position=None, parent=None):
        self.position = position
        self.parent = parent

        self.g=0
        self.h=0
        self.f=0

    def __eq__(self, other):
        return self.f == other.f

    def __lt__(self, other):
        return self.f < other.f
    
    def __gt__(self, other):
        return self.f > other.f

def isValid(position):
    x,y=position
    if(x >= 0 and y >= 0 and x < img.shape[0] and y < img.shape[1] and (img[position] != obstacle_color).any()):
        return True
    return False

def distance(a,b):
    dist=math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return dist

def getpath(current):
    while current is not None:
        img[current.position]=(112,25,25)
        current=current.parent

def dijkstra():
    start=Node(source,None)
    start.g=start.h=start.f=0

    dest=Node(end,None)
    dest.g=dest.h=dest.f=0

    q.put(start)
    dist[start.position[0],start.position[1]]=0

    while not q.empty():
        current=q.get()
        
        if current.f>dist[current.position[0],current.position[1]]:
            continue

        img[current.position]=(0, 140, 255)

        if current.position==dest.position:
            getpath(current)
            break

        for i in range(-1,2):
            for j in range(-1,2):
                pos=(current.position[0]+i,current.position[1]+j)
                if not isValid(pos):
                    continue
                
                child=Node(pos,current)
                child.g=current.g+distance(child.position,current.position)
                child.h=max(abs(child.position[0]-dest.position[0]),abs(child.position[1]-dest.position[1]))
                child.f=child.g+child.h
                
                if child.f<dist[pos[0],pos[1]]:                    
                    img[pos[0],pos[1]]=(255,0,0)
                    dist[pos[0],pos[1]]=child.f
                    q.put(child)

    return(current.g)

start_time=time.time()
print(dijkstra())
print("time: %s seconds" % (time.time() - start_time))
factor=10
upscale=np.full(((factor*m),(factor*n),3),255)

for i in range(m):
    for j in range(n):
        upscale[factor*i:(factor*i)+factor+1,factor*j:(factor*j)+factor+1]=img[i,j]

cv2.imshow("Path",upscale.astype(np.uint8))
cv2.waitKey(0)