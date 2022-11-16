# import the necessary packages
import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from scipy.spatial import distance as dist
import argparse
import imutils
import os

"""
Setting up variable Values
"""
MIN_CONF = 0.3
NMS_THRESH = 0.3
MIN_DISTANCE = 50

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Video successfully uploaded and displayed below')
		return render_template('output.html', filename=filename)

@app.route('/result/<filename>')
def resultant(filename):
	print("resultant working...")
	return Response(social_distance_detector(filename),mimetype='multipart/x-mixed-replace; boundary=frame')

def detect_people(frame, net, ln, personIdx=0):
	
	(H, W) = frame.shape[:2]
	results = []

	
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize lists of detected bounding boxes, centroids, and confidences
	boxes = []
	centroids = []
	confidences = []

	
	for output in layerOutputs:          # loop over each of the layer outputs
		for detection in output:         # loop over each of the detections
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personIdx and confidence > MIN_CONF:
			
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
	if len(idxs) > 0:
		for i in idxs.flatten():
	
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	return results

def social_distance_detector(filename):
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str, default="",
		help="path to (optional) input video file")
	ap.add_argument("-o", "--output", type=str, default="",
		help="path to (optional) output video file")
	ap.add_argument("-d", "--display", type=int, default=1,
		help="whether or not output frame should be displayed")

	args = vars(ap.parse_args(["--input", filename, "--output", "my_ouput.avi", "--display", "1"]))
	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join(["coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join(["yolov3.weights"])
	configPath = os.path.sep.join(["yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	if False:
		print("[INFO] setting preferable backend and target to CUDA...")
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream and pointer to output video file
	print("[INFO] accessing video stream...")
	print("file to be found at "+app.config['UPLOAD_FOLDER']+filename)
	vs = cv2.VideoCapture(app.config['UPLOAD_FOLDER']+filename) #capture from disk  
	
	writer = None

	while True:
		(grabbed, frame) = vs.read()    # read the next frame from the file
		if not grabbed:
			break
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))

		violate = set()

		if len(results) >= 2:
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")


			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					if D[i, j] < MIN_DISTANCE:
						violate.add(i)
						violate.add(j)

		for (i, (prob, bbox, centroid)) in enumerate(results):
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			if i in violate:
				color = (0, 0, 255)

			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

		text = "Social Distancing Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

		if args["display"] > 0:
			# show the output frame
			ret,buffer=cv2.imencode('.jpg',frame)
			op=buffer.tobytes()
			yield(b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + op + b'\r\n')

		if args["output"] != "" and writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 25,
				(frame.shape[1], frame.shape[0]), True)

		if writer is not None:
			writer.write(frame)


def social_distance_detector_fetch():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str, default="",
		help="path to (optional) input video file")
	ap.add_argument("-o", "--output", type=str, default="",
		help="path to (optional) output video file")
	ap.add_argument("-d", "--display", type=int, default=1,
		help="whether or not output frame should be displayed")

	args = vars(ap.parse_args(["--input", "", "--output", "my_ouput.avi", "--display", "1"]))
	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join(["coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join(["yolov3.weights"])
	configPath = os.path.sep.join(["yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


	if False:
		print("[INFO] setting preferable backend and target to CUDA...")
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	print("[INFO] accessing video stream...")
	vs = cv2.VideoCapture(0) #capture from camera
	
	writer = None

	while True:
		(grabbed, frame) = vs.read()
		if not grabbed:
			break
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))

		violate = set()

		if len(results) >= 2:
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					if D[i, j] < MIN_DISTANCE:
						violate.add(i)
						violate.add(j)

		for (i, (prob, bbox, centroid)) in enumerate(results):
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			if i in violate:
				color = (0, 0, 255)

			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

		text = "Social Distancing Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

		if args["display"] > 0:
			# show the output frame
			ret,buffer=cv2.imencode('.jpg',frame)
			op=buffer.tobytes()
			yield(b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + op + b'\r\n')

		if args["output"] != "" and writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 25,
				(frame.shape[1], frame.shape[0]), True)

		if writer is not None:
			writer.write(frame)

def generate_frames():
    while True:
        camera = cv2.VideoCapture()    
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/fetch')
def fetch():
    return render_template('read.html')

@app.route('/video')
def video():
    return Response(social_distance_detector_fetch(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
    
