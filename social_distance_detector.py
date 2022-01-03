# import the necessary packages
import social_distancing_config as config
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import easygui

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY,a,b,c,d
    mouseX = 0
    mouseY = 0
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #cv2.circle(image,(x,y),100,(255,0,0),-1)
        mouseX,mouseY = x,y
        #print(mouseX,mouseY)
        if not a:
            a=[x,y]
        elif not b:
            b=[x,y]
        elif not c:
            c=[x,y]
        elif not d:
            d=[x,y]

a=0
b=0
c=0
d=0
myvar=""
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(2)
writer = None

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# initialize the set of indexes that violate the minimum social
	# distance
	violate = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	M=np.array([])
	dst=np.array([])
	socialdistance=50
	if a!=0 and b!=0 and c!=0 and d!=0:
		size = myvar.split(',')
		size[0] = int(size[0])
		size[1] = int(size[1])
		x=0
		y=0
		if size[0]>=size[1]:
			x=1000
			y=1000*(size[1]/size[0])
			socialdistance=int(1000/size[0])
		else:
			y=1000
			x=1000*(size[0]/size[1])
			socialdistance=int(1000/size[1])
		x=int(x)
		y=int(y)
		print(x,y,socialdistance)
		pts1 = np.float32([a,b,c,d])
		pts2 = np.float32([[0,0],[y,0],[0,x],[y,x]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		dst = cv2.warpPerspective(frame,M,(y,x))
		#plt.imshow(dst)
		#plt.show()
		
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		feet=[]
		#move centroid to near person's feet
		for (i, (prob, bbox, centroid)) in enumerate(results):
			feet.append((centroid[0],bbox[3]))
		shoes=[]
		#perspective transform the coordinates
		matrix = M
		if M.any():
			for pair in feet:
				#nparr = np.array([int(pair[0]),int(pair[1])])
				#shoes=np.dot(M,nparr)
				p = (int(pair[0]),int(pair[1])) # your original point
				px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
				py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
				shoes.append(np.array([int(px), int(py)]))
			print(shoes)
			D = dist.cdist(shoes, shoes, metric="euclidean")
		else:
			D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				#if D[i, j] < config.MIN_DISTANCE:
				if D[i, j] < socialdistance:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = feet[i]
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		r= cX- startX
		cv2.circle(frame, (cX,endY), 5, color, 1)
		if dst.any():
			cv2.circle(dst, (shoes[i][0],shoes[i][1]), 5, color, 1)
		#cv2.circle(frame, (cX, cY), 5, color, 1)

	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2.putText(frame,"c-calibrate", (20,25), font, fontScale,(10,200,10),thickness,lineType)
		cv2.imshow("Frame", frame)
		if dst.any():
			cv2.imshow("Perspective Transform",dst)
		#Calbirate pause, and set points
		if cv2.waitKey(1) & 0xFF == ord('c'):
                    cv2.setMouseCallback('Frame',draw_circle)
                    while True:
                        print(a,b,c,d)
						#output texts
                        snapshot=frame.copy()
                        cv2.putText(snapshot,"r-reset", (20,45), font, fontScale,(10,200,10),thickness,lineType)
                        cv2.putText(snapshot,"Top Left=    "+str(a), (20,400), font, fontScale,(10,200,10),thickness,lineType)
                        cv2.putText(snapshot,"Top Right=    "+str(b), (20,420), font, fontScale,(10,200,10),thickness,lineType)
                        cv2.putText(snapshot,"Bottom Left=    "+str(c), (20,440), font, fontScale,(10,200,10),thickness,lineType)
                        cv2.putText(snapshot,"Bottom Right=    "+str(d), (20,460), font, fontScale,(10,200,10),thickness,lineType)
                        cv2.imshow('Frame', snapshot)
                        #if R is pressed, set all points to 0
                        if cv2.waitKey(1) & 0xFF == ord('r'):
                            a=0
                            b=0
                            c=0
                            d=0
						#if C is pressed again, unpause and ask for height and width
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            cv2.setMouseCallback('Frame',lambda *args:1+1)
                            if a!=0 and b!=0 and c!=0 and d!=0:
                                myvar = easygui.enterbox("Input height,width (height in topleft-botleft, width in topleft-topright)")
                                print(myvar)
                            break
		#key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		#if key == ord("q"):
		#	break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)

# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi