# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
#Imports
import os
import cv2
import math
import numpy as np
from pathlib import Path
from numpy.linalg import norm
import frameextractor as frames
from handshape_feature_extractor import HandShapeFeatureExtractor as features

#lists used throughout the code
train_list=list()
train_label=list()
test_list=list()
results_list=list()

#Extract single frame from the traing videos and then extract hand gesture feature
def extract_training_frames():
    #create Directory to hold 
    directory_name = "single_frames_train"
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

    train_directory = Path("traindata/")
    extract = features.get_instance()
    
    #extract the frames
    count=0
    for item in train_directory.iterdir():
        frames.frameExtractor(item,"single_frames_train/",count)
        count=count+1

    #extract and label the features from each training video
    count=0
    photo_directory = Path("single_frames_train/")
    for item in photo_directory.iterdir():
        new_name=item.name
        split_list = new_name.split('.')
        count_png=int(split_list[0])
        if(count_png==(count+1)):
            img = cv2.imread(item)
            a = np.array(img)
            grayscale_image_array = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            feature=extract.extract_feature(grayscale_image_array)
            train_list.append(feature)
            train_label.append(math.floor(count/3))
        count=count+1
     
    return None

#Extract single frame from the test videos and then extract hand gesture feature
def extract_test_frames():
    #Create directory to hold testing frames
    directory_name = "single_frames_test"
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

    test_directory = Path("test/")
    extract = features.get_instance()
    #extract the frames
    count=0
    for item in test_directory.iterdir():
        frames.frameExtractor(item,"single_frames_test/",count)
        count=count+1
    #extract the feature for each testing frame
    count=0
    photo_directory = Path("single_frames_test/")
    for item in photo_directory.iterdir():
        new_name=item.name
        split_list = new_name.split('.')
        count_png=int(split_list[0])
        if(count_png==(count+1)):
            img = cv2.imread(item)
            a = np.array(img)
            grayscale_image_array = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            feature=extract.extract_feature(grayscale_image_array)
            test_list.append(feature)
        count=count+1
        
        


    return None

#Use cosine similarity to assign the label o fthe highes tvalue to the testing features
def identify_gestures():
    
    for i in range(51):
        highest_label=0
        highest_val=0
        for j in range(51):
            vector_a = train_list[j]
            vector_b = test_list[i]
            cosine = np.dot(vector_a, vector_b.T) / (norm(vector_a) * norm(vector_b))
            if cosine>highest_val:
                highest_val=cosine
                highest_label=train_label[j]
        results_list.append(highest_label)
        print("Cosine Similarity to label:", highest_label," of ",highest_val)
    return None

def clean_results():
    #Correct Labeling issues
    for i in range(51):
        value=results_list[i]
        match value:
            case 0:
                results_list[i]=10
            case 1:
                results_list[i]=11
            case 2:
                results_list[i]=12
            case 3:
                results_list[i]=13
            case 4:
                results_list[i]=14
            case 5:
                results_list[i]=15
            case 6:
                results_list[i]=0
            case 7:
                results_list[i]=1
            case 8:
                results_list[i]=2
            case 9:
                results_list[i]=3
            case 10:
                results_list[i]=4
            case 11:
                results_list[i]=5
            case 12:
                results_list[i]=6
            case 13:
                results_list[i]=7
            case 14:
                results_list[i]=8
            case 15:
                results_list[i]=9
            case 16:
                results_list[i]=16
    
    #Convert list to numpy array
    numpy_results = np.array(results_list)

    #write the results to Results.csv
    with open('Results.csv', 'w') as results_file:
        NEWLINE_SIZE_IN_BYTES = 2
        np.savetxt(results_file, numpy_results, delimiter=",", fmt='%i')
        results_file.seek(0, os.SEEK_END) 
        results_file.seek(results_file.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
        results_file.truncate()
    return None


extract_training_frames()
extract_test_frames()
identify_gestures()
clean_results()