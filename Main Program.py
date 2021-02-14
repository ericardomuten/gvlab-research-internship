import numpy as np
import pandas as pd
import serial
import time
from keras.models import model_from_json
import nep


#PREPARATIONS STARTS FROM HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#LOAD MODEL = Preparing the trained models that want to be used
# load json and create model
json_file = open('D:\TUAT\GV Lab Research Internship\Machine Learning\Models\cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("D:\TUAT\GV Lab Research Internship\Machine Learning\Models\cnn_model_weight.hdf5")
print("Loaded model from disk")
#-----------------------------------------------------------------------------------------------------------------------

#NEP = the connection between python (recognition) and the matlab (servo executor)

node = nep.node("publisher_node") # Create a new node
conf = node.conf_pub(transport = "ZMQ", mode="many2one") # Select the configuration of the publisher
pub = node.new_pub("many2one_string", conf) # Set the topic and the configuration of the publisher
#-----------------------------------------------------------------------------------------------------------------------

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#CHANGEABLE VARIABLES
period = 0.02 #period of time for taking 1 frame of data (1/frequency)
num_of_data = 150 #number of data for each excel data file (after the normalization)
port = 'COM8'
#-----------------------------------------------------------------------------------------------------------------------

#MOVEMENTS DICTIONARY = Making dictionary for movements
movements = {
    0 : "gentle stroke",
    1 : "no touch",
    2 : "poke",
    3 : "press",
    4 : "scratch",
}

#PORT SETTING = Setting up the communication with the sensors
#(COM, baudrate, parity bit, stopbit, and data size for the serial port that connect to the sensor's board)
ser = serial.Serial(
    port=port,\
    baudrate=230400,\
    parity=serial.PARITY_NONE,\
    stopbits=serial.STOPBITS_ONE,\
    bytesize=serial.EIGHTBITS,\
        timeout=0)
print("connected to: " + ser.portstr)
ser.write(b'020100') #0201XX > set the sampling cycle, 00 for 4 ms and FA for 254 ms, we want as fast as possible = 00
ser.write(b'020301') #02030X > read from X sensors (start from 01 to 04)
ser.write(b'020201') #02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
time.sleep(0.05)
#port_sampling = "2001F70201007E020A01E701ED0062021400DD0064003800260021001E001D001C"  #input from manual (notepad)
#num_channel = int(int(port_sampling[0] + port_sampling[1], 16) / 8)  #determining number of active sensor's channel if input from manual
port_sampling = str(ser.readline())  #input from port
num_channel = int(int(port_sampling[2] + port_sampling[3], 16) / 8)  #determining number of active sensor's channel
#-----------------------------------------------------------------------------------------------------------------------

#DATA CONVERSION = Preparing the methods to convert port's data to normalized value of force
coefficient = np.zeros((3,4))
standard_value = np.zeros((4,1))
voltage = np.zeros((4,4))
T = np.zeros((4,1))
force = np.zeros((3,4))
max_voltage = 3.3 #maximum voltage known from the sensor's instruction manual
return_value_bit = 10 #number of return value bit known from the sensor's instruction manual
voltage_percentage = max_voltage/((2**return_value_bit)-1)
#filling a matrix consist of mean $ std for each axis for every sensors
name_of_coordinate = ["X", "Y", "Z"]
Mean = np.zeros((num_channel, 3)) #num_channel rows for each sensor, 3 columns for mean for 3 axis (x,y,z)
Std = np.zeros((num_channel, 3)) #num_channel rows for each sensor, 3 columns for mean for 3 axis (x,y,z)
for channel in range (0, num_channel):
    reference = pd.read_excel("D:/TUAT/GV Lab Research Internship/Machine Learning/Data/Cut_Data/Sensor" + str((num_channel+1)) + "/" + "Mean-Std_Dev_" + str(num_of_data) + ".xlsx", "Sheet")
    for axis in range (0, 3):
        Mean[channel, axis] = reference["Mean " + name_of_coordinate[axis]][0]
        Std[channel, axis] = reference["Std Dev " + name_of_coordinate[axis]][0]
#making coefficient and standard value matrix
file_read = pd.read_excel("D:/TUAT/GV Lab Research Internship/Machine Learning/Calibration Coefficient Sensor2.xlsx", "Sheet1")
#fill the coefficient matrix with numbers known from the sensor's instruction manual
for i in range (0, 3):
    for j in range (0, 4):
        coefficient[i, j] = file_read[j][i]
#fill the standard value matrix with numbers known from the sensor's instruction manual
for i in range (0, 4):
    standard_value[i, 0] = file_read[i][4]
#-----------------------------------------------------------------------------------------------------------------------

#CALIBRATION = preparing the starting offset & later on offset correction
#starting offset (PLEASE RUN THE "SENSOR CALIBRATION.PY" FIRST)
offset_file = pd.read_excel("D:/TUAT/GV Lab Research Internship/Machine Learning/" + "Offset.xlsx", "Sheet")
offset = np.zeros((num_channel, 3))
offset_column = ["Offset X", "Offset Y", "Offset Z"]
for channel in range (0, num_channel):
    for axis in range (0, 3):
        offset[channel, axis] = offset_file[offset_column[axis]][channel]

#later on offset
output = 0
#-----------------------------------------------------------------------------------------------------------------------

#RECOGNITION = preparing the variables needed for recognition

frame_check = 100 #check the recognition of motion every frame_check frames

#Preparing the windows
windows1 = np.zeros((num_of_data,3)) #windows for sensor 1 (channel 1)
windows2 = np.zeros((num_of_data,3)) #windows for sensor 2 (channel 2)
temporary = np.zeros((1,3)) #temporary matrix for new data

#PREPARATION DONE HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




#THE ONLINE PREDICTION PROGRAM STARTS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
count=0
#ser.write(b'020201')  # 02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
while True: #keep reading data from port, calculate, and send it to the model
    ser.write(b'020201')  # 02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
    time.sleep(period)  #period for taking 1 frame
    port_sampling = str(ser.readline())  # input from port
    port_sampling = port_sampling[2:(4+16*num_channel)] #cutting the excess b' and \r\n' in the read from port value

    #Converting data from port into force
    for channel in range(0, num_channel):
        for axis in range(0, 4):  # 3 axis + 1 Temperature
            voltage[channel, axis] = voltage_percentage * int(
                port_sampling[2 + 16 * channel + 4 * axis] + port_sampling[3 + 16 * channel + 4 * axis] + port_sampling[
                    4 + 16 * channel + 4 * axis] + port_sampling[5 + 16 * channel + 4 * axis], 16)  # sampling voltage
            voltage[channel, axis] = voltage[channel, axis] - standard_value[axis, 0]  # delta voltage
    for channel in range(0, num_channel):  # fill T with the value of delta voltage of Temperature
        T[channel, 0] = voltage[channel, 3]
    for channel in range(0, num_channel):
        for axis in range(0, 3):  # back to 3 axis because we don't need to calculate temperature anymore
            voltage[channel, axis] = voltage[channel, axis] - coefficient[axis, 3] * T[
                channel, 0]  # delta voltage acent
        voltage[channel, 3] = 0  # delete all the delta voltage of Temperature
    voltage = voltage.transpose()  # transpose the voltage matrix so we can do the matrix multiplication with the coefficient
    np.matmul(coefficient, voltage, force)  # the results is 3x4 matrix, the column represent each channel and the row represent each axis (Fx, Fy, Fz)

    #Normalizing the force data
    for channel in range (0, num_channel):
        for axis in range (0, 3):
            force[axis, channel] = force[axis, channel] - offset[channel, axis]
            force[axis, channel] = (force[axis, channel]-Mean[channel, axis])/Std[channel, axis]

    #Filling the windows with force data
    for channel in range(0, num_channel):
        #print(str(count)+str(': \t ')+"Fx"+": "+str(force[0, channel])+"\t"+"Fy"+": "+str(force[1, channel])+"\t"+"Fz"+": "+str(force[2, channel])+"\t"+"T: "+str(float(T[channel]+standard_value[3, 0]))+"\n") #print all the force+T
        if count<num_of_data: #fill the first num_of_data(40) data to the windows
            if channel == 0:
                for axis in range (0, 3): #3 axis (x,y,z)
                    windows1[count, axis] = force[axis, channel]
            else:
                for axis in range (0, 3): #3 axis (x,y,z)
                    windows2[count, axis] = force[axis, channel]
        else: #fill the next data
            if channel == 0:
                windows1 = windows1[1:num_of_data] #delete the oldest data
                for axis in range(0, 3):  # 3 axis (x,y,z) #fill the temporary matrix with new data
                    temporary[0, axis] = force[axis, channel]
                windows1 = np.concatenate((windows1, temporary)) #concatenate the windows with new data(temporary matrix)
            else:
                windows2 = windows2[1:num_of_data]  # delete the oldest data
                for axis in range(0, 3):  # 3 axis (x,y,z) #fill the temporary matrix with new data
                    temporary[0, axis] = force[axis, channel]
                windows2 = np.concatenate((windows2, temporary))  # concatenate the windows with new data(temporary matrix)

    #later on offset correction
    if output == 0 and count%3000 == 0 and count > 2999: #if the recognized movement is no touch then fill up the error matrix for later offset correction for every 3000 frames
        error = windows1
        offset = error.mean(0)

    #Starts to predict the movement made (recognition)
    if count > num_of_data and count%frame_check == 0:
        for channel in range(0, num_channel):
            if channel == 0:
                window = np.reshape(windows1, (-1, 1)) #reshapping the windows to match the model format of input
                window = np.expand_dims(window, 2) #reshapping the windows to match the model format of input
                window = np.transpose(window, [1, 0, 2]) #reshapping the windows to match the model format of input
                prediction = loaded_model.predict(window) #predict the movement
                output = np.argmax(prediction)
                print(movements.get(output), ", percentage = " + str(100 * prediction[0, output]))
                if output == 0:
                    pub.send_string(str(output)) #send the recognized motion number to the matlab
                    time.sleep(18)
                if output == 2:
                    pub.send_string(str(output)) #send the recognized motion number to the matlab
                    time.sleep(16)
                if output == 3:
                    pub.send_string(str(output)) #send the recognized motion number to the matlab
                    time.sleep(20)
                if output == 4:
                    pub.send_string(str(output)) #send the recognized motion number to the matlab
                    time.sleep(14)
            else:
                window = np.reshape(windows2, (-1, 1)) #reshapping the windows to match the model format of input
                window = np.expand_dims(window, 2) #reshapping the windows to match the model format of input
                window = np.transpose(window, [1, 0, 2]) #reshapping the windows to match the model format of input
                prediction = loaded_model.predict(window) #predict the movement
                output = np.argmax(prediction)
                print(movements.get(output), ", percentage = " + str(100 * prediction[0, output]))
                if output == 0:
                    pub.send_string(str(output))  # send the recognized motion number to the matlab
                    time.sleep(18)
                if output == 2:
                    pub.send_string(str(output))  # send the recognized motion number to the matlab
                    time.sleep(16)
                if output == 3:
                    pub.send_string(str(output))  # send the recognized motion number to the matlab
                    time.sleep(20)
                if output == 4:
                    pub.send_string(str(output))  # send the recognized motion number to the matlab
                    time.sleep(14)



    count = count + 1
    #ser.write(b'020201')  # 02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
ser.close() #closing the serial port

########################################################################################################################



# num_of_data = 40
# x = np.zeros((40,3))
# file_read = pd.read_excel("D:/TUAT/GV Lab Research Internship/Machine Learning/Data/Normalized_Cut_Data/Edo/40/poken_40_13.xlsx", "Sheet")
# for sample in range (0, num_of_data):
#     for coordinate in range (0, 3):
#         x[sample, coordinate] = file_read[coordinate][sample]
#
# x = np.reshape(x, (-1, 1))
# x = np.expand_dims(x, 2)
# x = np.transpose(x, [1, 0, 2])
#
# prediction = loaded_model.predict(x)
# output = np.argmax(prediction)
# print(prediction, 'probability')
# print(movements.get(output), ", percentage = " + str(100*prediction[0, output]))