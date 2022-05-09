import cv2
import numpy as np
import math

def pixels(image, parameters):
	slope, intercept = parameters
	y1 = image.shape[0]
	y2 = int((y1*2.9/4))
	if slope == 0:
		slope = 0.1
	x1 = int((y1 - intercept) / slope)
	x2 = int((y2 - intercept) / slope)
	return [[x1, y1, x2, y2]]

def regression_on_lines(image, lines):
	road_lines = []

	if lines is None:
		return road_lines

	left_line = []
	right_line = []

	w = image.shape[1]
	left_reg = (w*(1-(1/3)))
	rignt_reg = (w*(1/3))

	for line in lines:
		for x1, y1, x2, y2 in line:
			if x1 == x2:
				continue

			slope = (y2 - y1) / (x2 - x1)
			intercept = y1 - (slope * x1)

			if slope > 0:
				if x1 > rignt_reg and x2 > rignt_reg:
					right_line.append((slope, intercept))
			else:
				if x1 < left_reg and x2 < left_reg:
					left_line.append((slope, intercept))

	left_avg = np.average(left_line, axis=0)
	right_avg = np.average(right_line, axis=0)

	if len(left_line) > 0:
		road_lines.append(pixels(image, left_avg))

	if len(right_line) > 0:
		road_lines.append(pixels(image, right_avg))
	
	return road_lines

def filters(image):	
	temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	temp = cv2.GaussianBlur(temp, (3,3), 0)
	kernel = np.ones((2,2),np.uint8)
	temp = cv2.Canny(temp, 40, 130)
	temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
	return temp

def ROI(image):
	height, width = image.shape
	mask = np.zeros_like(image)
	triangles = np.array([
		[(int(width/4), height), (int(9*width/10), height), (int(width/2), int(13*height/20))]], np.int32)
	cv2.fillPoly(mask, triangles, 255)
	return mask

def display_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 3, cv2.LINE_AA)
	line_image = cv2.addWeighted(frame, 1, line_image, 1, 1) # lines on original image
	return line_image

def prediction_turn(image, reg_lines):
	height, width, _ = image.shape

	length_reg_l = len(reg_lines)
	y_move_axis = int(height/2)

	if length_reg_l == 2:
		_, _, left_a, _ = reg_lines[0][0]
		_, _, right_a, _ = reg_lines[1][0]
		middle = width/2
		x_move_axis = (left_a + right_a)/2-middle
	elif length_reg_l == 1:
		x1, _, x2, _ = reg_lines[0][0]
		x_move_axis = x2 - x1
	elif length_reg_l == 0:
		x_move_axis = 0
	
	radian_angle = math.atan(x_move_axis / y_move_axis)
	angle_mid = int((radian_angle * 180)/np.pi)

	angle_of_move = angle_mid +90
	#print("angle: ", angle_of_move)
	return angle_of_move		


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480)) # you can change height and width if you want

cap = cv2.VideoCapture('Texas.mp4')
while(cap.isOpened()):
	ret, frame = cap.read()
	
	if ret is False:
		break
	
	frame = cv2.resize(frame, (640, 480)) # you can change height and width if you want
	height, width = frame.shape[0], frame.shape[1]
	
	timer = cv2.getTickCount()
	
	canny_image = filters(frame) # Canny Edge Detected image
	roi_mask = ROI(canny_image) # ROI masked image
	canny_image = cv2.bitwise_and(canny_image, roi_mask) # ROI-masked-canny image

	lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 50, np.array([]), minLineLength=75, maxLineGap=100) # HoughLinesProbabilistic
	regression_lines = regression_on_lines(np.copy(frame), lines)
	line_image = display_lines(frame, regression_lines) # Lines on the zero_like image
	fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
	turn_prediction = prediction_turn(frame, regression_lines)
	
	cv2.putText(line_image, 'FPS: ' + str(int(fps)), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
	if turn_prediction <= 90:
		cv2.putText(line_image, 'Turn Left', (15,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
	elif turn_prediction >= 95:
		cv2.putText(line_image, 'Turn Right', (15,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
	cv2.imshow('Original', line_image)

	out.write(line_image)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
cap.release()
out.release()
cv2.destroyAllWindows()