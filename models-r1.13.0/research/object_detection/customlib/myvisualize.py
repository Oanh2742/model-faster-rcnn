import cv2
import numpy as np
import math

list_color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
				(128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,0,128),(0,128,128)]
list_class = ['CAR', 'BUS', 'TRUCK', 'CONTAINER']

class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


# def drawPredected(_frame,_objectID,_centroid):
	

def drawBox(_frame, _class, _box ):
	[startY, startX, endY, endX] = _box
	cv2.rectangle(_frame, (int(startX), int(startY)), (int(endX), int(endY)), 
		list_color[int(_class)-1], 2) 
	return _frame

def drawTracking(_frame, _objectID, _centroid):
	text = ".{}".format(_objectID)
	if len(_centroid.shape) == 1: 
		cv2.putText(_frame, text, (_centroid[0]+10, _centroid[1]+10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, list_color[_objectID%12], 4)
	else:
		cv2.putText(_frame, text, (_centroid[-1][0], _centroid[-1][1]+10),
			cv2.FONT_HERSHEY_SIMPLEX, 1, list_color[_objectID%12], 4)
	return _frame

def drawOrbit(_frame, _objectID, _centroid):
	if len(_centroid.shape) == 1:
		return _frame
	else:
		for i in range(1,len(_centroid)):
			cv2.line(_frame, (_centroid[i-1][0], _centroid[i-1][1]),
				(_centroid[i][0], _centroid[i][1]), list_color[_objectID%12], 6)
	return _frame

def drawDataOnBoard(_frame, _objectID, _centroid, _classes, _centroids, _row):
	if len(_centroid.shape) == 1:
		if (_centroid[0],_centroid[1]) in _centroids:
			i = _centroids.index((_centroid[0],_centroid[1]))
			vehicle_type = list_class[int(_classes[i]) - 1]
		else: 
			vehicle_type = "UNKWOWN"
	else:
		if (_centroid[-1][0],_centroid[-1][1]) in _centroids:
			i = _centroids.index((_centroid[-1][0],_centroid[-1][1]))
			vehicle_type = list_class[int(_classes[i]) - 1]
		else:
			vehicle_type = "UNKWOWN"
	strID = str(_objectID).ljust(8)
	strtype = vehicle_type.ljust(8)
	data = "{0}{1}".format(strID,strtype)
	cv2.putText(_frame, data, (10,120 + _row*25) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
	return _frame

def drawVelocity(_frame, _objectID, _centroid, _speed):
	v = "{} km/h".format(_speed)
	if len(_centroid.shape) == 1:
		cv2.putText(_frame, v, (_centroid[0]+10, _centroid[1]+10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4)
	else:
		cv2.putText(_frame, v, (_centroid[-1][0], _centroid[-1][1]+10),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4)
	return _frame


def drawSystem1(_frame, _objects, _classes, _centroids, _passed):
	# cửa sổ phụ để biểu diễn các thông tin theo dõi 
	# sub_frame = np.zeros((720,320,3), np.uint8)
	# tiêu đề của hệ thống 

	logo = "Embedded Networking Lab - HUST"
	cv2.putText(_frame, logo, (_frame.shape[1]-600,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),4)
	
	# tittle = "VEHICLES TRACKING SYSTEM"
	# cv2.putText(sub_frame, tittle, (55,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
	# số xe đang trong theo dõi  
	tracked = "CURRENTLY TRACKING: {}".format(len(_centroids))
	cv2.putText(_frame, tracked, (20, _frame.shape[0]-10) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)
	# tổng số xe đã đi qua 
	total = "TOTAL PASSED: {}".format(_passed)
	cv2.putText(_frame, total, (int(_frame.shape[1]/2), _frame.shape[0]-10) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)
	# bảng dữ liệu theo dõi 
	# tracking_board = "ID      TYPE    VELOCITY(m/s)"
	# cv2.putText(sub_frame, tracking_board, (10,95) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
	# gắn dữ liệu theo dõi lên bảng 
	#row = 0

	for (objectID, centroid) in _objects.items():
		_frame = drawTracking(_frame, objectID, centroid)
		_frame = drawOrbit(_frame, objectID, centroid)
		#sub_frame = drawDataOnBoard(sub_frame, objectID, centroid, _classes, _centroids, row)
		#row += 1
	cv2.line(_frame, (500,int(_frame.shape[0]*0.1)),(900,int(_frame.shape[0]*0.1)),
		(0,255,0), 4)		

	cv2.line(_frame, (0,int(_frame.shape[0]*0.7)),(_frame.shape[1],int(_frame.shape[0]*0.7)),
		(0,0,255), 4)

	#_frame = cv2.resize(_frame,(960,720))
	# create full frame	
	#_frame = np.hstack((_frame,sub_frame))
	return _frame

def drawSystem2(_frame, _objects, _speeds):
	logo = "Embedded Networking Lab - HUST"
	cv2.putText(_frame, logo, (_frame.shape[1]-600,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),4)

	for (objectID, centroid) in _objects.items():
		_frame = drawTracking(_frame, objectID, centroid)

		# speed = _speeds[objectID]
		# _frame = drawVelocity(_frame, objectID, centroid, speed)
	return _frame