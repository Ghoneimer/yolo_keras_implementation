import numpy as np
import csv
#read the file
data_path = 'D:/last_desktop/car_detection/object_with_occlusion/' #the path for the folder that contain all images and files.
csv_path = 'example5.CSV' #excel sheet file name.
def load_dataset(file_path):
    dataset_corners = []
    dataset_midpoints = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                dataset_corners.append({'frameId' : line[0], 
                                'Xmin' : float(line[1])/1920., #Xmin, Ymin is the bottom left normalized coordinates
                                'Ymin' : float(line[2])/1200., 
                                'Xmax' : float(line[3])/1920., #Xmax, Ymax is the upper right normalized coordinates
                                'Ymax' : float(line[4])/1200., 
                                'ocluded' : line[5], 
                                'label' : line[6]})
                dataset_midpoints.append({'frameId' : line[0], 
                                'Xc' : ((float(line[1])/1920.)+(float(line[3])/1920.))/2, #Xc, Yc is the normalized center point
                                'Yc' : ((float(line[2])/1200.)+(float(line[4])/1200.))/2, 
                                'h' : (float(line[4])/1200.)-(float(line[2])/1200.), #normalized hieght of the box
                                'w' : (float(line[3])/1920.)-(float(line[1])/1920.), #normalized width of the box
                                'label' : line[6]})
            except:
                continue # some images throw error during loading 
    return dataset_corners, dataset_midpoints
dataset_corners, dataset_midpoints = load_dataset(data_path+csv_path)

class Box():
    def __init__(self, x, y, w, h, iD):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.iD = iD

# make a list of all boxes objects for k-mean algorithm.
boxes = []
for i in range (len(dataset_midpoints)):
    boxes.append(Box(0, # put all the boxes at the origin to make the iou distances 
                     0, # depend only on h, w when run the k-mean algorithm
                     dataset_midpoints[i]['h'], 
                     dataset_midpoints[i]['w'],
                     i))

# Computes the overlap of two boxes on one axis
# x1 is the center of box1 on this axis
# len1 is the length of box1 on this axis
# x2 is the center of box2 on this axis
# len2 is the length of box2 on this axis
# The return value is the length of overlap on this axis
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# Compute the intersection of box a and box b
# a and b are examples of type Box
# The return value area is the intersection of box a and box b
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# Compute the union of box a and box b
# a and b are examples of type Box
# The return value u is the union area of box a and box b
# Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# Calculate iou for box a and box b
# a and b are examples of type Box
# The return value is iou for box a and box b
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# Initialize centroids with k-means ++ to reduce the impact of randomly initialized centroids on the final result
# boxes is a list of all Box objects bounding boxes
# n_anchors is the k value of k-means (number of clusters)
# The return value centroids is the initialized n_anchors centroid
def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)

    #choos random box object from all the boxes and put it in centroids list
    centroid_index = np.random.choice(boxes_num, 1) #choose randomly a number from 0 to boxes_num
    centroids.append(boxes[centroid_index[0]]) #centroids list now have only one boxe object 

    print(centroids[0].w,centroids[0].h)

    # loop from centroid_index = 0 to n_anchors-1
    # after this loop all the centroids will be initialized.
    for centroid_index in range(0,n_anchors-1): 

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        # loop over all the boxes.
        for box in boxes:
            min_distance = 1
            # enumerate(centroids) produce a tuple with index and value 
            # in centroids so centroid_i is the index and centroid is the value and the loop iterate over all ellements
            for centroid_i, centroid in enumerate(centroids):  
                distance = (1 - box_iou(box, centroid)) 
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break

    return centroids


# Perform k-means to calculate new centroids
# boxes is a list of all Box objects bounding boxes
# n_anchors is the k value of k-means
# centroids is the center of all the clusters
# Return value new_centroids is the calculated new cluster center
# Return values groups is a list of boxes contained by n_anchors clusters
# The return value loss is the sum of the distances of the nearest centroids to which all box distances belong
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    #initialize the new centroids and the groups list which contain n_anchors list contain boxes
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0,0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


# Calculate the centroids for the n_anchors bounding boxes
# n_anchors is the number of anchors
# loss_convergence is the smallest change in allowed loss
# grid_size * grid_size is the number of rasters
# iterations_num is the maximum number of iterations
# plus = 1 enable k means ++ to initialize the centroids
def compute_centroids(boxes,n_anchors,loss_convergence,grid_size,iterations_num,plus):

    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        print("loss %i = %f" %(iterations-1, loss))
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            centroids = centroids
            groups = groups
            loss = loss  
            break
        old_loss = loss

        for centroid in centroids:
            print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    for centroid in centroids:
        print("k-means result：\n")
        print(centroid.w * grid_size, centroid.h * grid_size)
    return centroids, groups, loss

centroids, groups, loss = compute_centroids(boxes,n_anchors = 5,
                                            loss_convergence = 1e-6,
                                            grid_size = 19,
                                            iterations_num = 100,
                                            plus = 1)
'''
# determine the anchor box for every object from the 93086 objects 
# by generating rl list that will be used in the gen_yolo_truth code
rl = np.zeros((len(dataset_midpoints),1))
for j in range (len(groups[0])):
    rl[groups[0][j].iD] = 0
for j in range (len(groups[1])):
    rl[groups[1][j].iD] = 1
for j in range (len(groups[0])):    
    rl[groups[2][j].iD] = 2
for j in range (len(groups[0])):
    rl[groups[3][j].iD] = 3
for j in range (len(groups[0])):
    rl[groups[4][j].iD] = 4
#save the list as numpy array
np.save('D:/last_desktop/car_detection/YOLO implementation tensorflow/anchorList', rl)
'''