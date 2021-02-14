# -*-# coding: utf-8 -*-

"""
This modele provides eneric loading, showing and saving of input data.

This includes some uncompleted function such as ...
"""

# import
import os
import numpy as np
import scipy
import sklearn as sk
import tensorflow as ts
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import subplot2grid, tight_layout, subplots_adjust
from scipy.interpolate import interp1d

import pylab

# custom import
# from lowpass import butter_lowpass, butter_lowpass_filter

segment_points = []

def onclick(event):

    # if event.key == 'z':

    print('event.button=%d, event.x=%d, event.y=%d, event.xdata=%f, \
    event.ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
    global segment_points
    segment_point = int(round(event.xdata))
    segment_points.append(segment_point)
    plt.plot(event.xdata, event.ydata, 'o', color='r', markersize=5)
    # fig.canvas.draw()


class Segment_data:

    def __init__(self, fileName, subject_name):

        self.subject = subject_name
        #self.cwd = os.getcwd()
        self.cwd = 'D:\\TUAT\\GV Lab Research Internship\\Machine Learning' #this one is weird, ask Tomoya
        #CHANGE THIS TO CHANGE THE DIRECTORY
        self.dataPath = 'D:\\TUAT\\GV Lab Research Internship\\Machine Learning' + '\\Data\\Data_Raw\\Sensor1\\' + self.subject + '\\'
        self.leftPath = fileName + '.csv'
        self.filename = fileName

        self.left = pd.read_csv(self.dataPath + self.leftPath)
        # self.left.drop(self.left.columns[len(self.left.columns)-1], axis=1, inplace=True)
#YOU CHANGE THIS FROM 3:6 TO ....
        self.left = self.left.values[:, 3:6]

        # self.time = self.left['Times']

        # self.left.drop(['Times'], axis=1, inplace=True)
        # self.left_columns = self.left.columns

        # self.left = self.left.as_matrix()

        # self.sp_list = pd.read_excel('Segment_indices_list2.xlsx')
        # self.sp_list = self.sp_list.as_matrix()
        # self.sp_name = self.sp_list[:, 0]
        # self.ind1_list = self.sp_list[:, 1]
        # self.ind2_list = self.sp_list[:, 2]

        #CHANGE HERE FOR CHANGING THE DIRECTORY
        self.savesp_path = self.cwd + '\\Data\\segment_points\\Sensor1\\' + self.subject + '\\'

    def load_index_list(self, index):

        self.ind1 = self.ind1_list[index]

        self.ind1 = int(self.ind1)

        print(self.sp_name[index], 'file name')
        print(self.ind1, 'ind1')
# CHANGE THIS ONE!
    def calc_distance(self):

        '''
        self.ind1 = ind1
        self.ind2 = ind2
        '''

        self.distance_left = np.zeros([self.left.shape[0], 2])
        #self.distance_left = np.linalg.norm(self.left[:,0]*self.left[:,0] + self.left[:,1]*self.left[:,1] + self.left[:,2]*self.left[:,2], axis=1)

        self.distance_left = np.linalg.norm(self.left[:, self.ind1*3-3:self.ind1*3]-self.left[:, self.ind2*3-3:self.ind2*3], axis=1)

        self.distance_left_filter = butter_lowpass_filter(self.distance_left, 2, 60)

    def plot_raw_data(self):

        plt.title(self.filename + '_left')
        plt.xlabel('Time')
        plt.ylabel('Position [mm]')
        self.time_raw = [i for i in range(0, self.left.shape[0])]
        plt.plot(self.time_raw, self.left[:, 0])
        plt.show()
        plt.close()

    def show_input(self, save_name):

        fig = plt.figure()
        # ax = fig.add_subplot(111, projection = '3d')
        ax = Axes3D(fig)

        plt.plt.ion()
        column = list(self.left.columns.values)
        l = range(0, self.left.shape[0])
        for t in l[0::10]:  # book1.shape[0]
            plt.pyplot.cla()
            for i in range(1, int(self.left.shape[1] / 3)):
                ax.set_xlabel("X-axis")
                ax.set_xlabel("Y-axis")
                ax.set_xlabel("Z-axis")
                # plt.xlim(xmin, xmax)
                # plt.ylim(ymin, ymax)
                # plt.zlim(zmin, zmax)
                x1 = self.left[t, i * 3 - 3]
                y1 = self.left[t, i * 3 - 2]
                z1 = self.left[t, i * 3 - 1]
                ax.scatter(x1, y1, z1, s=25, color='blue', edgecolor='blue')
                ax.text(x1, y1, z1, '%s' % (column[i * 3 - 2]), size=10)

            plt.pyplot.pause(0.05)

            plt.savefig(save_name + str(t) +".png")
            plt.pause(0.05)

    def plot_raw_distance(self):

        fig = plt.figure()
        plt.subplot(121)
        plt.title(self.filename + '_distance_left')
        plt.xlabel('Time')
        plt.ylabel('Distance [mm]')
        self.time = [i for i in range(0, self.left.shape[0])]
        plt.plot(self.time, self.left)
        plt.plot(self.time, self.left_filter)
        plt.show()
        plt.close()
#CHANGE THIS ONE!!
    def splitout_manual(self):

        self.segment_point = []
        self.time = [i for i in range(0, self.left.shape[0])]

        fig = plt.figure()
        plt.xlabel('Time')
        plt.ylabel('Force [N]')

        #plt.plot(self.time, self.distance_left[:, 0], 'r')  # filtered value
        plt.plot(self.time, self.left[:, 0], 'r')    # filtered value
        plt.plot(self.time, self.left[:, 1], 'b')    # filtered value
        plt.plot(self.time, self.left[:, 2], 'k')    # filtered value

        # plt.plot(self.time, self.left_filter, 'k')  # raw value

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show

    def plot_segment(self):

        np.savetxt(self.savesp_path + self.filename + '_segment_points.txt', self.segment_points, delimiter=',')

        self.segment_middle = int(self.segment_points.shape[0]//2)
        self.segment_start = self.segment_points[:self.segment_middle]
        self.segment_end = self.segment_points[self.segment_middle:]
        fig = plt.figure(figsize=(25, 12))
        plt.xlabel('Time')
        plt.ylabel('Distance [mm]')
        plt.plot(self.time, self.left[:, 0], 'r')  # filtered value
        plt.plot(self.time, self.left[:, 1], 'b')  # raw value
        plt.plot(self.time, self.left[:, 2], 'k')  # raw value

        plt.scatter(self.segment_start, self.left[self.segment_start, 0], marker='o', c='green', s=80)
        plt.scatter(self.segment_end, self.left[self.segment_end, 0], marker='*', c='blue', s=80)

        plt.show()
        # plt.savefig(self.savefig_path + self.filename + '_SP.png')
        # plt.close()

    def cut_and_spline(self, size = 100):

        self.nCycle = int(len(segment_points) / 2)
        self.nJoint = self.left.shape[1]

        xmin = min(self.left[:, self.ind1])
        xmax = max(self.left[:, self.ind1])
        xmin_L = min(self.left[:, self.ind1])
        xmax_L = max(self.left[:, self.ind1])


        # global reference
        fig1 = plt.figure(figsize=(18, 12))
        ax1 = subplot2grid((3, self.nCycle), (0, 0), colspan=self.nCycle)
        ax1.set_title(self.filename + '_right')
        ax1.plot(self.time_raw, self.right[:, self.ind1])
        ax1.scatter(self.segment_start, self.right[self.segment_start, self.ind1], marker='o', c='green', s=70)
        ax1.scatter(self.segment_end, self.right[self.segment_end, self.ind1], marker='*', c='black', s=70)
        
        ax1.plot(self.time_raw, self.left[:, self.ind1+1])
        ax1.plot(self.time_raw, self.left[:, self.ind1+2])


        movement_left, movement_right = np.zeros((self.nCycle, size, self.nJoint)), np.zeros((self.nCycle, size, self.nJoint))
        movement_left_L, movement_right_L = np.zeros((self.nCycle, size, self.nJoint)), np.zeros((self.nCycle, size, self.nJoint))


        for a in range(0, self.nCycle):
            start = segment_points[a] + 1
            end = segment_points[a + self.nCycle] + 1
            length = end - start
            for i in range(0, self.nJoint):
                self.temp1_left = self.left[start:end, i]
                x = np.linspace(0, 1, length)
                x_new = np.linspace(0, 1, size)
                self.temp2_left = interp1d(x, self.temp1_left)(x_new)
                movement_left[a, :, i] = self.temp2_left

                self.temp1_right = self.right[start:end, i]
                x = np.linspace(0, 1, length)
                x_new = np.linspace(0, 1, size)
                self.temp2_right = interp1d(x, self.temp1_right)(x_new)
                movement_right[a, :, i] = self.temp2_right

                self.time_temp1 = [j for j in range(0, self.temp1_right.shape[0])]
                self.time_temp2 = [j for j in range(0, self.temp2_right.shape[0])]


                if i == self.ind1:
                    ax1 = subplot2grid((3, self.nCycle), (1, a), sharey=ax1)
                    ax1.set_ylim(xmin, xmax)
                    ax1.plot(self.time_temp1, self.temp1_right, 'blue')
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.05)

                    if a == 0:
                        plt.setp(ax1.get_yticklabels(), visible=True)
                    else:
                        plt.setp(ax1.get_yticklabels(), visible=False)

                if i == self.ind1:
                    ax1 = subplot2grid((3, self.nCycle), (2, a))
                    ax1.set_ylim(xmin, xmax)
                    ax1.plot(self.time_temp2, self.temp2_right, 'blue')
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.05)

                    if a == 0:
                        plt.setp(ax1.get_yticklabels(), visible=True)
                    else:
                        plt.setp(ax1.get_yticklabels(), visible=False)

        plt.show()
        plt.savefig(self.savefig_path + self.filename + '_all_' + str(size) + '.png')
        plt.close()



        # local reference
        fig2 = plt.figure(figsize=(18, 12))
        ax1 = subplot2grid((3, self.nCycle), (0, 0), colspan=self.nCycle)
        ax1.set_title(self.filename + '_right_L')
        ax1.plot(self.time_raw, self.right_L[:, self.ind1])
        ax1.scatter(self.segment_start, self.right_L[self.segment_start, self.ind1], marker='o', c='green', s=70)
        ax1.scatter(self.segment_end, self.right_L[self.segment_end, self.ind1], marker='*', c='black', s=70)



        for a in range(0, self.nCycle):
            start = segment_points[a] + 1
            end = segment_points[a + self.nCycle] + 1
            length = end - start
            for i in range(0, self.nJoint):
                self.temp1_left_L = self.left_L[start:end, i]
                x = np.linspace(0, 1, length)
                x_new = np.linspace(0, 1, size)
                self.temp2_left_L = interp1d(x, self.temp1_left_L)(x_new)
                movement_left_L[a, :, i] = self.temp2_left_L

                self.temp1_right_L = self.right_L[start:end, i]
                x = np.linspace(0, 1, length)
                x_new = np.linspace(0, 1, size)
                self.temp2_right_L = interp1d(x, self.temp1_right_L)(x_new)
                movement_right_L[a, :, i] = self.temp2_right_L

                self.time_temp1_L = [j for j in range(0, self.temp1_right_L.shape[0])]
                self.time_temp2_L = [j for j in range(0, self.temp2_right_L.shape[0])]



                if i == self.ind1:
                    ax1 = subplot2grid((3, self.nCycle), (1, a), sharey=ax1)
                    ax1.set_ylim(xmin_L, xmax_L)
                    ax1.plot(self.time_temp1_L, self.temp1_right_L, 'blue')
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.05)

                    if a == 0:
                        plt.setp(ax1.get_yticklabels(), visible=True)
                    else:
                        plt.setp(ax1.get_yticklabels(), visible=False)

                if i == self.ind1:
                    ax1 = subplot2grid((3, self.nCycle), (2, a))
                    ax1.set_ylim(xmin_L, xmax_L)
                    ax1.plot(self.time_temp2_L, self.temp2_right_L, 'blue')
                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.05)

                    if a == 0:
                        plt.setp(ax1.get_yticklabels(), visible=True)
                    else:
                        plt.setp(ax1.get_yticklabels(), visible=False)

        plt.show()
        plt.savefig(self.savefig_path + self.filename + '_all_L_' + str(size) + '.png')
        plt.close()

    def plot_temp(self, dataPath, fileName, ind1, ind2):

        self.dataPath = dataPath
        self.filename = fileName
        self.ind1 = ind1
        self.ind2 = ind2

        self.raw1 = pd.read_csv(self.dataPath + self.filename + '_30.csv')
        self.raw2 = pd.read_csv(self.dataPath + self.filename + '_40.csv')
        self.raw3 = pd.read_csv(self.dataPath + self.filename + '_50.csv')
        self.raw4 = pd.read_csv(self.dataPath + self.filename + '_60.csv')
        self.raw5 = pd.read_csv(self.dataPath + self.filename + '_70.csv')
        self.raw6 = pd.read_csv(self.dataPath + self.filename + '_80.csv')
        self.raw7 = pd.read_csv(self.dataPath + self.filename + '_90.csv')
        self.raw8 = pd.read_csv(self.dataPath + self.filename + '_100.csv')
        self.raw1 = self.raw1.as_matrix()
        self.raw2 = self.raw2.as_matrix()
        self.raw3 = self.raw3.as_matrix()
        self.raw4 = self.raw4.as_matrix()
        self.raw5 = self.raw5.as_matrix()
        self.raw6 = self.raw6.as_matrix()
        self.raw7 = self.raw7.as_matrix()
        self.raw8 = self.raw8.as_matrix()

        self.time1 = [j for j in range(1, 31)]
        self.time2 = [j for j in range(1, 41)]
        self.time3 = [j for j in range(1, 51)]
        self.time4 = [j for j in range(1, 61)]
        self.time5 = [j for j in range(1, 71)]
        self.time6 = [j for j in range(1, 81)]
        self.time7 = [j for j in range(1, 91)]
        self.time8 = [j for j in range(1, 101)]

        fig = plt.figure()
        plt.xlabel('Time')
        plt.ylabel('Position [mm]')
        plt.subplot(121)
        plt.plot(self.time1, self.raw1[0:30, self.ind1*3-3])
        plt.plot(self.time2, self.raw2[0:40, self.ind1*3-3])
        plt.plot(self.time3, self.raw3[0:50, self.ind1*3-3])
        plt.plot(self.time4, self.raw4[0:60, self.ind1*3-3])
        plt.plot(self.time5, self.raw5[0:70, self.ind1*3-3])
        plt.plot(self.time6, self.raw6[0:80, self.ind1*3-3])
        plt.plot(self.time7, self.raw7[0:90, self.ind1*3-3])
        plt.plot(self.time8, self.raw8[0:100, self.ind1*3-3])
        plt.plot(self.time1, self.raw1[0:30, self.ind1*3-3], label='30')
        plt.plot(self.time2, self.raw2[0:40, self.ind1*3-3], label='40')
        plt.plot(self.time3, self.raw3[0:50, self.ind1*3-3], label='50')
        plt.plot(self.time4, self.raw4[0:60, self.ind1*3-3], label='60')
        plt.plot(self.time5, self.raw5[0:70, self.ind1*3-3], label='70')
        plt.plot(self.time6, self.raw6[0:80, self.ind1*3-3], label='80')
        plt.plot(self.time7, self.raw7[0:90, self.ind1*3-3], label='90')
        plt.plot(self.time8, self.raw8[0:100, self.ind1*3-3], label='100')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode='expand')
        plt.xlim(0, 100)

        plt.subplot(122)
        plt.plot(self.time1, self.raw1[0:30, self.ind1*3-3+82])
        plt.plot(self.time2, self.raw2[0:40, self.ind1*3-3+82])
        plt.plot(self.time3, self.raw3[0:50, self.ind1*3-3+82])
        plt.plot(self.time4, self.raw4[0:60, self.ind1*3-3+82])
        plt.plot(self.time5, self.raw5[0:70, self.ind1*3-3+82])
        plt.plot(self.time6, self.raw6[0:80, self.ind1*3-3+82])
        plt.plot(self.time7, self.raw7[0:90, self.ind1*3-3+82])
        plt.plot(self.time8, self.raw8[0:100, self.ind1*3-3+82])
        plt.plot(self.time1, self.raw1[0:30, self.ind1*3-3+82], label='30')
        plt.plot(self.time2, self.raw2[0:40, self.ind1*3-3+82], label='40')
        plt.plot(self.time3, self.raw3[0:50, self.ind1*3-3+82], label='50')
        plt.plot(self.time4, self.raw4[0:60, self.ind1*3-3+82], label='60')
        plt.plot(self.time5, self.raw5[0:70, self.ind1*3-3+82], label='70')
        plt.plot(self.time6, self.raw6[0:80, self.ind1*3-3+82], label='80')
        plt.plot(self.time7, self.raw7[0:90, self.ind1*3-3+82], label='90')
        plt.plot(self.time8, self.raw8[0:100, self.ind1*3-3+82], label='100')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode='expand')
        plt.xlim(0, 100)

        plt.savefig(self.filename + '_all_splined.png')
        plt.show()
#1ST RUN FROM HERE UNTIL THE TOP, DONT FORGET TO CHANGE THE DIRECTORY FOR FILE READ (datapath) AND FOR THE SAVING DIRECTORY OF THE SEGMENT POINTS(savesp)

# names = []
# f = open('sign_list2.txt')
# names = f.read().splitlines()


#2ND RUN HERE UNTIL SPLITOUT
segment_points = []
index = 0
class1 = Segment_data('notouch', 'Takuya')
# class1.load_index_list(index)
# class1.calc_distance()
# class1.plot_raw_data()
# class1.plot_raw_distance()
class1.splitout_manual()

#3RD RUN HERE UNTIL BOTTOM
class1.segment_points = np.array(segment_points)


class1.plot_segment()
print('-----end')
# print(index+1, 'is next index')
# print(class1.sp_name[index + 1], 'is next sign')



self = class1