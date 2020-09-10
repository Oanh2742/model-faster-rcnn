from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import math

# class KalmanFilter:

#     kf = cv2.KalmanFilter(4, 2)
#     kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
#     kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

#     def Estimate(self, coordX, coordY):
#         ''' This function estimates the position of the object'''
#         measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
#         self.kf.correct(measured)
#         predicted = self.kf.predict()
#         return predicted


class CentroidTracker():
	def __init__(self, maxDisappeared=5):
		# khởi tạo ID duy nhất cho đối tượng tiếp theo cùng với 2 từ
		# điển được sử dụng để theo dõi các ánh xạ của ID vật thể với 
		# tâm của nó và số khung hình liên tiếp mà nó được đánh dấu là   
		# biến mất tương ứng 
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.speeds = OrderedDict()
		self.disappeared = OrderedDict()
		self.rectangle = OrderedDict()
		# self.KalmanCentroid = OrderedDict()
		# lưu trữ số khung hình liên tiếp tối đa mà một vật thể được
		# đánh dấu là "biến mất" tới khi chúng ta hủy nó khỏi việc theo
		# dõi
		self.maxDisappeared = maxDisappeared

	def register(self,rect, centroid):
		# khi đăng kí theo dõi một vật thể, chúng ta sử dụng ID đang tồn
		# tại để lưu trữ tâm
		self.objects[self.nextObjectID] = centroid
		self.speeds[self.nextObjectID] = 0
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1
		self.rectangle[self.nextObjectID] = rect
		# self.KalmanCentroid[self.nextObjectID] = KalmanFilter()
	def deregister(self, objectID):
		# để hủy theo dõi một vật thể, chúng ta xóa ID của nó khỏi tất
		# cả các từ điển 
		del self.objects[objectID]
		del self.disappeared[objectID]
		del self.speeds[objectID]
		del self.rectangle[objectID]
		# del self.KalmanCentroid[objectID]
	def update(self, rects, upSpeed):
		# kiểm trả nếu danh sách các hộp giới hạn đầu vào là rỗng 
		if len(rects) == 0:
			# lặp qua tất cả các vật thể đang bị theo dõi và đánh dấu
			# chúng là biến mất
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# Nếu đạt tới giá trị tối đa của số khung hình liên 
				# tiếp mà một object bị đánh dấu là biến mất, hủy đăng 
				# kí nó
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# trả về kết quả không có tâm hoặc thông tin theo dõi để 
			# cập nhật 
			return self.objects,self.rectangle,self.speeds
		
		# khởi tạo mảng các vật thể đầu vào cho khung hiện tại 
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# lặp qua các hộp giới hạn 
		for (i, (startY, startX, endY, endX)) in enumerate(rects):
			# sử dụng tọa độ của hộp giới hạn để xác định tâm 
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
		# nếu chúng ta đang không theo dõi vật thể nào, lấy các tâm đầu 
		# vào và đăng ký cho chúng s
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(rects[i],inputCentroids[i])

		# ngược lại, chúng ta đang theo dõi một số vất thể thì chúng ta
		# cần khớp các tâm đầu vào với tâm của các vật thể đang có. 
		else:
			# lấy ra  danh sách các ID đối tượng và tâm tương ứng 
			objectIDs = list(self.objects.keys())
			objectOrbits = list(self.objects.values())
			objectCentroids = []
			for i in objectOrbits:
				if len(i.shape) == 1 :
					objectCentroids.append(i)
				else:
					objectCentroids.append(i[-1]) 

			# tính toán khoảng cách giữa mỗi cặp tâm của các vật thể
			# với tâm của các đầu vào, mục đích của chúng ta là khớp tâm 
			# của một đầu vào với tâm của một vật thể đang có   
			D = dist.cdist(objectCentroids, inputCentroids)

			# để thực hiện việc khớp, chúng ta phải tìm giá trị nhỏ nhất
			# trong mỗi hàng, sau đó sắp xếp các chỉ mục hàng dựa vào giá 
			# trị nhỏ nhất của chúng sao cho hàng có giá trị nhỏ nhất nằm 
			# ở đầu của danh sách chỉ mục 
			rows = D.min(axis=1).argsort()

			# tiếp theo, thực hiện quy trình tương tự trên các cột bằng cách 
			# tìm kiếm gía trị nhỏ nhẩ trên mỗi cột và sau đó sắp xếp chúng
			# dựa vào danh sách chỉ mục hàng đã tính toán trước đó  
			cols = D.argmin(axis=1)[rows]

			# để xác đính xem chúng ta cần cập nhật, đăng ký hoặc hủy đăng 
			# ký  một vật thể, chúng ta cần giữ việc theo dõi 
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				# self.objects[objectID] = inputCentroids[col]
				self.objects[objectID] = np.vstack((self.objects[objectID], inputCentroids[col]))
				self.rectangle[objectID] = rects[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids isx
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(rects[col],inputCentroids[col])
		
		if upSpeed == 0:
			for ID, centro in self.objects.items():
				if len(centro.shape) == 1:
					self.speeds[ID] = 0
				else:
					if (centro.shape[0] == 10):
						x = centro[-1][0] -centro[-10][0]
						y = centro[-1][1] -centro[-10][1]
					else:
						x = centro[-1][0] -centro[-2][0]
						y = centro[-1][1] -centro[-2][1]
					v = int(math.sqrt(x*x + y*y) * 59 / 1920 * 60 * 3.6)
					self.speeds[ID] = v

		# return the set of trackable objects
		return self.objects,self.rectangle ,self.speeds
