# Face-Recognition-Based-Attendance-System
Python Version : 3.7

Libraries Used : 
Opencv
Numpy
Pandas
PIL
Kivy
Shutil

Abstract: 
  To detect real time human face are used and a simple fast Principal Component Analysis has used to recognize the faces detected with a high accuracy rate. The matched face is used to mark attendance of the students. Our system maintains the attendance records of students automatically.Our module enrolls the student’s face. This enrolling is a onetime process and their face will be stored in the database. During enrolling of face we require a system since it is a onetime process. You can have your own roll number as your id which will be unique for each student. The presence of each student will be updated in a database. Attendance is marked after student identification. 

Algorithm :
	Step 1: Take an image of a normal expression of a human face.
	Step 2: Converts the color image to grayscale.
	Step 3: Crop the 3 facial image region of interest (ROI) (eyes, eye brows and lip) from the image by defining region.
	Step 4: Find edges of all image region.
	Step 5: Take an  image of a faces and create dataset
	Step 7: Train the dataset.
	Step 8: Recognize the faces in video camera.
 Step 9: Mark the attendance accordingly.

