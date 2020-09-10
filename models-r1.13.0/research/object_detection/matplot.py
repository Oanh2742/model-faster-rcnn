import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

measurement_x = []
measurement_y = []
predict_x = []
predict_y = []
n_measurement = []
n_predict = []
fname = "data.txt"
count = 0
line = 0
with open(fname, 'r') as P:
    for _line in P:
    	count += 1
    	if (count %4==0):
	    	n_measurement.append(count/4)
	    	n_predict.append(count/4+1)	
print("Total number of lines is:", count)
file = open ("data.txt","r")
for i in range(count) :
	line += 1
	if (line % 16 == 1) :
		measurement_x.append(float(file.readline().replace('\n','')))
	if (line % 16 == 2) :
		measurement_y.append(float(file.readline().replace('\n','')))
	if (line % 16 == 3) :
		predict_x.append(float(file.readline().replace('[','').replace(']','').replace('\n','')))
	if (line % 16 == 4) :
		predict_y.append(float(file.readline().replace('[','').replace(']','').replace('\n','')))
print("predict x "+str(predict_x))
print("predict y "+str(predict_y))
print("measurement x "+str(measurement_x))
print("measurement y "+str(measurement_y))
plt.plot(measurement_x,measurement_y,'bo')
# for x,y,z in zip(measurement_x,measurement_y,n_measurement):

#     label = "{:.2f}".format(z)

#     plt.annotate(label, # this is the text
#                  (x,y), # this is the point to label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,10), # distance from text to points (x,y)
#                  ha='center') # horizontal alignment can be left, right or center

plt.plot(predict_x,predict_y,'r-')
# for x,y,z in zip(predict_x,predict_y,n_predict):

#     label = "{:.2f}".format(z)

#     plt.annotate(label, # this is the text
#                  (x,y), # this is the point to label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,10), # distance from text to points (x,y)
#                  ha='center') # horizontal alignment can be left, right or center
# plt.plot(measurement_x,measurement_y,'bo',label="Measurement")
# plt.plot(predict_x,predict_y,'ro',label="Predict")

plt.show()
file.close()