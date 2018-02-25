import csv
import numpy as np
from random import *


data_path = 'D:/last_desktop/car_detection/object_with_occlusion/' #the path for the folder that contain all images and files.
csv_path = 'example5.CSV' #excel sheet file name.


def load_dataset(file_path):
    dataset = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                dataset.append({'id':line[0], 'xmin': float(line[1]) ,'ymin': float(line[2]),'xmax': float(line[3]),'ymax': float(line[4]),'class':line[6]})
            except:
                continue # some images throw error during loading 
    return dataset
	
z=load_dataset(data_path+csv_path)
print(z[1])
print(len(z))

def prepare_and_normalize(data_set):
    z=data_set
    Data_set=[]
    photos=[]
    for i in range(0,len(z)):
        photos.append([z[i]['id']])
        if (z[i]['class']=="car"):
            Data_set.append([float(1), float((((z[i]['xmax']-z[i]['xmin'])/2)+z[i]['xmin'])/1920), float((((z[i]['ymax']-z[i]['ymin'])/2)+z[i]['ymin'])/1200), float((z[i]['xmax']-z[i]['xmin'])/1920), float((z[i]['ymax']-z[i]['ymin'])/1200), float(1), float(0), float(0), float(0), float(0), float(0)])
        elif (z[i]['class']=="pedestrian"):
            Data_set.append([float(1), float((((z[i]['xmax']-z[i]['xmin'])/2)+z[i]['xmin'])/1920), float((((z[i]['ymax']-z[i]['ymin'])/2)+z[i]['ymin'])/1200), float((z[i]['xmax']-z[i]['xmin'])/1920), float((z[i]['ymax']-z[i]['ymin'])/1200), float(0), float(1), float(0), float(0), float(0), float(0)])
        elif (z[i]['class']=="Red"):
            Data_set.append([float(1), float((((z[i]['xmax']-z[i]['xmin'])/2)+z[i]['xmin'])/1920), float((((z[i]['ymax']-z[i]['ymin'])/2)+z[i]['ymin'])/1200), float((z[i]['xmax']-z[i]['xmin'])/1920), float((z[i]['ymax']-z[i]['ymin'])/1200), float(0), float(0), float(1), float(0), float(0), float(0)])
        elif (z[i]['class']=="Green"):
            Data_set.append([float(1), float((((z[i]['xmax']-z[i]['xmin'])/2)+z[i]['xmin'])/1920), float((((z[i]['ymax']-z[i]['ymin'])/2)+z[i]['ymin'])/1200), float((z[i]['xmax']-z[i]['xmin'])/1920), float((z[i]['ymax']-z[i]['ymin'])/1200), float(0), float(0), float(0), float(1), float(0), float(0)])
        elif (z[i]['class']=="Yellow"):
            Data_set.append([float(1), float((((z[i]['xmax']-z[i]['xmin'])/2)+z[i]['xmin'])/1920), float((((z[i]['ymax']-z[i]['ymin'])/2)+z[i]['ymin'])/1200), float((z[i]['xmax']-z[i]['xmin'])/1920), float((z[i]['ymax']-z[i]['ymin'])/1200), float(0), float(0), float(0), float(0), float(1), float(0)])
        elif (z[i]['class']=="trafficLight"):
            Data_set.append([float(1), float((((z[i]['xmax']-z[i]['xmin'])/2)+z[i]['xmin'])/1920), float((((z[i]['ymax']-z[i]['ymin'])/2)+z[i]['ymin'])/1200), float((z[i]['xmax']-z[i]['xmin'])/1920), float((z[i]['ymax']-z[i]['ymin'])/1200), float(0), float(0), float(0), float(0), float(0), float(1)])
    return Data_set,photos

y,l= prepare_and_normalize(z)
print(y[1])
print(len(y))
print(l[1])
print(len(l))

def photo_numbering(data_set):
    z=data_set
    lp=z[0]
    photo_no=[]
    n=0
    for i in range(0,len(z)):
        if(z[i]!=lp):
            n+=1
            lp=z[i]
        else:
            n=n
        photo_no.append(n)
    return photo_no

p=photo_numbering(l)
last=p[-1]+1
print(p[-1])
print(len(p))
        


def grid_selector(data_set):
    z=data_set
    grids=[]
    grid_no=[]
    for i in range(0,len(z)):
        ix=0
        iy=0
        while(z[i][1]>ix/19):
            ix+=1
        while(z[i][2]>iy/19):
            iy+=1
        grids.append([ix-1,iy-1])
    for o in range(0,len(grids)):
        n=(grids[o][1]+1)*(grids[o][0]+1)+((19-(grids[o][1]+1))*((grids[o][0]+1)-1))
        grid_no.append(n-1)        
    return grids

v=grid_selector(y)
print(v[52784:52793])
print(len(v))
'''
#generating this list randomly for test purposes
rl=[]
for g in range(0,len(v)):
    r=randint(0,4)
    rl.append(r)
'''
rl = np.load('D:/last_desktop/car_detection/YOLO implementation tensorflow/anchorList.npy')
def create(photo_num, grid_num, data, place):
    final_dataset=np.zeros((photo_num[-1]+1,20,20,5,11))
    for i, m in enumerate(photo_num):
        final_dataset[m][grid_num[i][0]][grid_num[i][1]][int(place[i])] = data[i]
    return final_dataset



k=p[0:10]
print(k)
f=create(p,v,y,rl)
print(f[7552][13][9])














    
