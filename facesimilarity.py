
import  tensorflow as tf
from align import detect_face
import facenet
import cv2
import imutils
import numpy as np
import argparse
import face_recognition
from matplotlib.pyplot import imshow,show

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.Session()
facenet.load_model("20180408-102900/20180408-102900.pb")

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    # print(feed_dict)
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def getFace(img):
	img = imutils.resize(img,width=1000)
	face_locations = face_recognition.face_locations(img)
	if len(face_locations)>0:
		print("found more than one image")
		return None
	elif len(face_locations)==0:
		print("no face found in the image")
		return None
	else:
		top, right, bottom, left = face_locations[0]
		face_image = img[top:bottom, left:right]
		resized = cv2.resize(face_image, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
		embedding = getEmbedding(facenet.prewhiten(resized))
		return embedding


def compare2face(img1,img2):
    face1 = getFace(img1)
    face2 = getFace(img2)
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(face1, face2)))
        return dist
    return -1

threshold_same=[]
def setthreshold_from_training_data(img1,img2):
	"""Set the threshold for similarity by averaging the similarity from training data"""
	distance=compare2face(img1,img2)
	if distance!=-1:
		threshold_same.append(distance) 
		
threshold=np.median(threshold_same)
"""
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
distance = compare2face(img1, img2)
if distance<threshold:
	return "same"
else:
	return "Not Same"
"""

@app.route("/scoreimages",methods=['GET','POST'])
def score_images():
	if request.method=='POST':

		if len(request.files)>1:
			if 'base_image' in request.files:
				base_image=face_recognition.load_image_file(request.files['base_image'])
			else:
				print("INPUT IMAGE NOT FOUND")
				return -1
			if 'new_image' in request.files:
				base_image=face_recognition.load_image_file(request.files['new_image'])
			else:
				print("INPUT IMAGE NOT FOUND")
				return -1
			distance = compare2face(img1, img2)
			if distance<threshold:
				return "same"
			else:
				return "Not Same"

		else:
			print("WRONG INPUT NO IMAGE FILES GIVEN")
			return -1