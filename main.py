# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
from asyncio.windows_events import NULL
import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from handshape_feature_extractor import HandShapeFeatureExtractor as features
import frameextractor as frames
import math
import pandas as pd
from numpy.linalg import norm
from pathlib import Path
from PIL import Image
import csv
## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
rows, cols = (51, 2)
train_arr = [[0]*cols]*rows
test_arr = [[0]*cols]*rows
train_list=list()
train_label=list()
test_list=list()
results_list=list()


def extract_training_frames():
    
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

    count=0
    for item in train_directory.iterdir():
        frames.frameExtractor(item,"single_frames_train/",count)
        count=count+1

    count=0
    photo_directory = Path("single_frames_train/")
    for item in photo_directory.iterdir():
        new_name=item.name
        split_list = new_name.split('.')
        count_png=int(split_list[0])
        if(count_png==(count+1)):
            img = Image.open(item)
            print(item.name)
            a = np.array(img)
            grayscale_image_array = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            feature=extract.extract_feature(grayscale_image_array)
            train_list.append(feature)
            train_label.append(math.floor(count/3))
        count=count+1
     
    return None

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
def extract_test_frames():
    directory_name = "single_frames_test"
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

    train_directory = Path("test/")
    extract = features.get_instance()

    count=0
    for item in train_directory.iterdir():
        frames.frameExtractor(item,"single_frames_test/",count)
        count=count+1

    count=0
    photo_directory = Path("single_frames_test/")
    for item in photo_directory.iterdir():
        new_name=item.name
        split_list = new_name.split('.')
        count_png=int(split_list[0])
        if(count_png==(count+1)):
            img = Image.open(item)
            print(item.name)
            a = np.array(img)
            grayscale_image_array = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            feature=extract.extract_feature(grayscale_image_array)
            test_arr[count][0]=math.ceil(count/3)
            test_list.append(feature)
        count=count+1
        
        


    return None

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
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
    #for i in results_list:
        #print(type(i))

    # Write to a CSV file
    numpy_results = np.array(results_list)

    with open('Results.csv', 'w') as results_file:
        NEWLINE_SIZE_IN_BYTES = 2 # 2 on Windows?
        np.savetxt(results_file, numpy_results, delimiter=",", fmt='%i')
        results_file.seek(0, os.SEEK_END) # Go to the end of the file.
        # Go backwards one byte from the end of the file.
        results_file.seek(results_file.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
        results_file.truncate() # Truncate the file to this point.
    
        
    #print(len(numpy_results))
    #np.savetxt('Results.csv', numpy_results, delimiter=",", fmt='%i')

    return None


extract_training_frames()
extract_test_frames()
identify_gestures()
clean_results()