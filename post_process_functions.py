# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:25:15 2017

@author: Tiger
"""

from skimage.morphology import skeletonize
from skimage.morphology import *
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import mahotas as mah
import cv2
import numpy as np
from skimage import measure
import csv
from PIL import Image
import pickle as pickle

from data_functions import *
from plot_functions import *
from UNet import *


""" defines a cell object for saving output """
class Cell:
    def __init__(self, num):
        self.num = num
        self.fibers = []    # creates a new empty list for each cell
        self.coords = np.zeros([1, 2], dtype=int)

    def add_fiber(self, fibers):
        self.fibers.append(fibers)


    def add_coords(self, new_coords):
        self.coords = np.append(self.coords, new_coords, axis=0)

""" Instead of getting minimum intensity:
        1) loop through all of the masked overlapping regions
          and find out ALL of the UNIQUE values within the NON-overlapped regions
        2) use these unqique values to index the list of cells
        3) find the number of fibers associated with each of these cells, and assign it to the cell with the MOST 
           number of fibers ALREADY associated and that has AT LEAST more than 1 fiber???
           if all only have 1 fiber, then just assign to the one that has the minium intensity???
"""
def sort_max_fibers(masked, list_M_cells):
    """ maybe add a step where you cycle through and get all the indexes of cells WITH fibers
        so don't have to waste time later looping over cells that don't even have fibers        
    """
    idx_cells = []
    for T in range(len(list_M_cells)):
        fibers = list_M_cells[T].fibers
        if fibers:
            idx_cells.append(T)
    
    import operator    
    binary_masked = masked > 0
    labelled = measure.label(binary_masked)
    cc_overlap = measure.regionprops(labelled, intensity_image=masked)
    sort_mask = np.zeros(masked.shape)

    for M in range(len(cc_overlap)): 
        
        overlap_coords = cc_overlap[M]['coords']
        
        cells_overlap = []
        all_numFibers = []
        for T in range(len(idx_cells)):   
           idx = idx_cells[T]
                  
           fiber_coords = list_M_cells[idx].coords 
           fibers = list_M_cells[idx].fibers 
           
           combined = np.append(overlap_coords, fiber_coords, axis=0)
           orig_len = len(combined)
           
           """ find if fiber overlaps by seeing if there is anything unique 
               RATHER than actually seeing if every pixel matches
           """
           uniq = np.unique(combined, axis=0)
           if len(uniq) < orig_len:
               cells_overlap.append(idx)
               all_numFibers.append(len(fibers))                      
                          
        if len(cells_overlap) > 1:
            cell_index, value = max(enumerate(all_numFibers), key=operator.itemgetter(1))
            
            """ (4) set the entire region to be of value cell_index """    
            for T in range(len(overlap_coords)):
               sort_mask[overlap_coords[T,0], overlap_coords[T,1]] = cells_overlap[cell_index]        
        
        print('Tested: %d overlapped of total: %d' %(M, len(cc_overlap)))     

    return sort_mask


"""
Find branch point in example image.
"""
def find_branch_points(sk):
    branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
    branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    br1=mah.morph.hitmiss(sk,branch1)
    br2=mah.morph.hitmiss(sk,branch2)
    br3=mah.morph.hitmiss(sk,branch3)
    br4=mah.morph.hitmiss(sk,branch4)
    br5=mah.morph.hitmiss(sk,branch5)
    br6=mah.morph.hitmiss(sk,branch6)
    br7=mah.morph.hitmiss(sk,branch7)
    br8=mah.morph.hitmiss(sk,branch8)
    br9=mah.morph.hitmiss(sk,branch9)    
    br=br1+br2+br3+br4+br5+br6+br7+br8+br9
    return br



#""" re-runs all the VALIDATION IMAGES to modify the fiber threshold"""
#def rerun_VALIDATION():    
#    input_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Testing/Valid_fibers/'
#    all_names = read_file_names(input_path)   
#    for T in range(len(all_csv)):
#        all_fibers_im = readIm_counter(DAPI_path,all_names,T)
#        all_fibers = np.asarray(all_fibers_im, dtype=float)          
#        #all_fibers = load_pkl(input_path, all_csv[T])
#        skeletonize_all_fibers(all_fibers, T, DAPI_tmp = np.zeros([8208,8208]), minLength=25, minLengthSingle=150)    


""" re-runs all the outputs to modify the fiber threshold"""
def rerun_all():    
    input_path = './SPATIAL_W_301000_Laminin_PDL/'
    all_csv = read_file_names(input_path)   
    for T in range(len(all_csv)):
        all_fibers = load_pkl(input_path, all_csv[T])
        
        if T < 5:
            add = 11
        elif T < 10: 
            add = 21 - 5

        skeletonize_all_fibers(all_fibers, T + add, DAPI_tmp = np.zeros([8208,8208]), minLength=18, minLengthSingle=72)    

""" Read and combine csv into single files containing lengths, numsheaths, ect...

***NEED TO FIX ==> when row is empty, still must add empty slot!!!

 """
def read_and_comb_csv_as_SINGLES():
    all_fibers = []
    all_numCells = []
    all_numShea = []
    all_numMFLC = []
    
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    input_path = filedialog.askdirectory(parent=root, initialdir="D:/Tiger/AI stuff/RESULTS/",
                                    title='Please select input directory')
    input_path = input_path + '/'

    all_csv = read_file_names(input_path)
    first = 1;
    output_name = all_csv[0] 
    output_name = output_name.split('.')[0]
    
    
    with open('Results_' + output_name + '_num_sheaths.csv', 'w') as sheaths:
        with open('Results_' +  output_name + '_lengths.csv', 'w') as lengths:
            with open('Results_' + output_name + '_cells.csv', 'w') as cells:
               with open('Results_' + output_name + '_mFLC.csv', 'w') as mFLC:

                    for T in range(len(all_csv)):
                        
                        filename = all_csv[T]
                        empty = 0
                        with open(input_path + filename, 'r') as csvfile:
                            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                            counter = 0
                        
                            for row in spamreader:
                                print(', '.join(row))
                                
                                row = list(filter(lambda a: a != '[]', row))
                                
                                
                                if counter % 2 != 0:
                                    counter = counter + 1
                                    continue
                                
                                for t in range(len(row)):
                                    if row[t] == '[]' or not row[t] :
                                        continue
                                    row[t] =  float(row[t])

                                if counter == 0:   all_fibers.append(row); wr = csv.writer(lengths, quoting=csv.QUOTE_ALL); wr.writerow(all_fibers[0]);

                                elif counter == 2: 
                                    all_numCells.append(row[0]);   # append the Num Ensheathed 
                                elif counter == 8:
                                    all_numCells.append(row[0]);   # append the Num MBP+
                                elif counter == 10:
                                    all_numCells.append(row[0]);   # append the Num Cells   
                                    wr = csv.writer(cells, quoting=csv.QUOTE_ALL); 
                                    wr.writerow(all_numCells);
                                    all_numCells = []

                                elif counter == 4: all_numShea.append(row); wr = csv.writer(sheaths, quoting=csv.QUOTE_ALL); wr.writerow(all_numShea[0]);

                                elif counter == 6: all_numMFLC.append(row); wr = csv.writer(mFLC, quoting=csv.QUOTE_ALL); wr.writerow(all_numMFLC[0]);


                                all_fibers = []
                                #all_numCells = []
                                all_numShea = []
                                all_numMFLC = []

                                
                                if counter == 10:
                                    break
                                counter = counter + 1
                            
                        if not empty:
                            first = 0    

""" Read and combine csv """
def read_and_comb_csv():
    all_fibers = []
    all_numCells = []
    all_numShea = []
    all_numMFLC = []
    
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    input_path = filedialog.askdirectory(parent=root, initialdir="D:/Tiger/AI stuff/RESULTS/",
                                    title='Please select input directory')
    input_path = input_path + '/'

    all_csv = read_file_names(input_path)
    output_name = 'Combined' + '_' + all_csv[0]    
    first = 1;
    for T in range(len(all_csv)):
        
        filename = all_csv[T]
        empty = 0
        with open(input_path + filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            counter = 0
        
            for row in spamreader:
                print(', '.join(row))
                
                row = list(filter(lambda a: a != '[]', row))
                
                if first == 1 and not row and counter == 0:
                    print('skip')
                    empty = 1
                    break
                
                if counter % 2 != 0 or not row:
                    counter = counter + 1
                    continue
                
                for t in range(len(row)):
                    if row[t] == '[]' or not row[t] :
                        continue
                    row[t] =  float(row[t])
                if first:
                    if row[0] and counter == 0:   all_fibers.append(row)
                    elif row[0] and counter == 2: all_numCells.append(row)
                    elif row[0] and counter == 4: all_numShea.append(row)
                    elif row[0] and counter == 6: all_numMFLC.append(row)
                elif not first and row:
                    if row[0] and counter == 0:   all_fibers[0].extend(row)
                    elif row[0] and counter == 2: all_numCells[0].extend(row)
                    elif row[0] and counter == 4: all_numShea[0].extend(row)
                    elif row[0] and counter == 6: all_numMFLC[0].extend(row)                                
                                
                if counter == 6:
                    break
                counter = counter + 1
            
        if not empty:
            first = 0    
    #l = [all_fibers[0], all_numCells[0], all_numShea[0], all_numMFLC[0]]
        
    with open(output_name, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(all_fibers[0])
        wr.writerow(all_numCells[0])
        wr.writerow(all_numShea[0])
        wr.writerow(all_numMFLC[0])    
        
        
""" Read and combine csv """
def read_and_comb_csv_ALL_TOGETHER_4_FILES():
    all_fibers = []
    all_numCells = []
    all_numShea = []
    all_numMFLC = []
    
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    input_path = filedialog.askdirectory(parent=root, initialdir="/Users/Neuroimmunology Unit/Anaconda3/AI stuff/MyelinUNet/Source/",
                                    title='Please select input directory')
    input_path = input_path + '/'
       

    all_csv = read_file_names(input_path)    
    with open('all_lengths' + all_csv[0], 'w') as lengths:
        with open('all_EnsheathCells' + all_csv[0], 'w') as ensheathed:
            with open('all_NumSheaths' + all_csv[0], 'w') as numSheaths:
                with open('all_mFLC' + all_csv[0], 'w') as mFLC:
    
                    for T in range(len(all_csv)):
                        
                        filename = all_csv[T]
                        empty = 0
                        with open(input_path + filename, 'r') as csvfile:
                            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                            counter = 1
                        
                            for row in spamreader:
                                print(', '.join(row))
                                
                                row = list(filter(lambda a: a != '[]', row))
                                
#                                if not row:
#                                    print('skip')
#                                    empty = 1
#                                    break
                                                    
                                if counter == 1:
                                    wr = csv.writer(lengths, quoting=csv.QUOTE_ALL)
                                    wr.writerow(row)
            
                                if counter == 3:
                                    wr = csv.writer(ensheathed, quoting=csv.QUOTE_ALL)
                                    wr.writerow(row)
                                    
                                if counter == 5:
                                    wr = csv.writer(numSheaths, quoting=csv.QUOTE_ALL)
                                    wr.writerow(row)
                                    
                                if counter == 7:
                                    wr = csv.writer(mFLC, quoting=csv.QUOTE_ALL)
                                    wr.writerow(row)
            
                                counter = counter + 1
        
""" Read and combine csv """
def read_and_comb_csv_16():       
    fold_nam = 'uFNet-5_CSVs/'
    input_path = './' + fold_nam 

    all_csv = read_file_names(input_path)
    
    X = 0
    while X < len(all_csv):

        output_name = 'Combined' + '_' + all_csv[X]    
        first = 1;
        all_fibers = []
        all_numCells = []
        all_numShea = []
        all_numMFLC = []
        print(output_name)
        for T in range(16):
            filename = all_csv[X + T]
            empty = 0
            with open(input_path + filename, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                counter = 0
            
                for row in spamreader:
                    #print(', '.join(row))
                    
                    row = list(filter(lambda a: a != '[]', row))
                    
                    if first == 1 and not row and counter == 0:
                        print('skip')
                        empty = 1
                        break
                    
                    if counter % 2 != 0 or not row:
                        counter = counter + 1
                        continue
                    
                    for t in range(len(row)):
                        if row[t] == '[]' or not row[t] :
                            continue
                        row[t] =  float(row[t])
                    if first:
                        if row[0] and counter == 0:   all_fibers.append(row)
                        elif row[0] and counter == 2: all_numCells.append(row)
                        elif row[0] and counter == 4: all_numShea.append(row)
                        elif row[0] and counter == 6: all_numMFLC.append(row)
                    elif not first and row:
                        if row[0] and counter == 0:   all_fibers[0].extend(row)
                        elif row[0] and counter == 2: all_numCells[0].extend(row)
                        elif row[0] and counter == 4: all_numShea[0].extend(row)
                        elif row[0] and counter == 6: all_numMFLC[0].extend(row)                                
                                    
                    if counter == 6:
                        break
                    counter = counter + 1
                
            if not empty:
                first = 0    
        #l = [all_fibers[0], all_numCells[0], all_numShea[0], all_numMFLC[0]]
            
        with open(output_name, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            if all_fibers:
                wr.writerow(all_fibers[0])
                wr.writerow(all_numCells[0])
                wr.writerow(all_numShea[0])
                wr.writerow(all_numMFLC[0])    
            
        X = X + 16
        
        
        
""" Read and combine csv """
def read_and_comb_csv_doubles():       
    fold_nam = 'uFNet-5_CSVs/1) doubles/'
    input_path = './' + fold_nam 

    all_csv = read_file_names(input_path)
    
    X = 0
    count = 0
    while X < len(all_csv):
        output_name = 'Combined' + '_' + all_csv[X]    
        first = 1;
        all_fibers = []
        all_numCells = []
        all_numShea = []
        all_numMFLC = []
        print(output_name)
        for T in range(2):
            filename = all_csv[X + T * 3]
            empty = 0
            with open(input_path + filename, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                counter = 0
            
                for row in spamreader:
                    #print(', '.join(row))
                    
                    row = list(filter(lambda a: a != '[]', row))
                    
                    if first == 1 and not row and counter == 0:
                        print('skip')
                        empty = 1
                        break
                    
                    if counter % 2 != 0 or not row:
                        counter = counter + 1
                        continue
                    
                    for t in range(len(row)):
                        if row[t] == '[]' or not row[t] :
                            continue
                        row[t] =  float(row[t])
                    if first or not all_fibers or not all_numCells or not all_numShea or not all_numMFLC:
                        if row[0] and counter == 0:   all_fibers.append(row)
                        elif row[0] and counter == 2: all_numCells.append(row)
                        elif row[0] and counter == 4: all_numShea.append(row)
                        elif row[0] and counter == 6: all_numMFLC.append(row)
                    elif not first and row:
                        if row[0] and counter == 0:   all_fibers[0].extend(row)
                        elif row[0] and counter == 2: all_numCells[0].extend(row)
                        elif row[0] and counter == 4: all_numShea[0].extend(row)
                        elif row[0] and counter == 6: all_numMFLC[0].extend(row)                                
                                    
                    if counter == 6:
                        break
                    counter = counter + 1
                
            if not empty:
                first = 0    
        #l = [all_fibers[0], all_numCells[0], all_numShea[0], all_numMFLC[0]]
            
        with open(output_name, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            if all_fibers:
                wr.writerow(all_fibers[0])
                wr.writerow(all_numCells[0])
                wr.writerow(all_numShea[0])
                wr.writerow(all_numMFLC[0])    
            
        if count == 2:
            count = 0
            X = X + 5
        else:
            X = X + 1
            count = count + 1
    

""" Read and combine csv """
def read_and_comb_csv_duplicates():       
    fold_nam = 'uFNet-5_CSVs/2) combined_doubles/'
    input_path = './' + fold_nam 

    all_csv = read_file_names(input_path)
    
    X = 0
    count = 0
    while X < len(all_csv):
        output_name = 'Combined' + '_' + all_csv[X]    
        first = 1;
        all_fibers = []
        all_numCells = []
        all_numShea = []
        all_numMFLC = []
        print(output_name)
        for T in range(2):
            filename = all_csv[X + T * 3]
            empty = 0
            with open(input_path + filename, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                counter = 0
            
                for row in spamreader:
                    #print(', '.join(row))
                    
                    row = list(filter(lambda a: a != '[]', row))
                    
                    if first == 1 and not row and counter == 0:
                        print('skip')
                        empty = 1
                        break
                    
                    if counter % 2 != 0 or not row:
                        counter = counter + 1
                        continue
                    
                    for t in range(len(row)):
                        if row[t] == '[]' or not row[t] :
                            continue
                        row[t] =  float(row[t])
                    if first or not all_fibers or not all_numCells or not all_numShea or not all_numMFLC:
                        if row[0] and counter == 0:   all_fibers.append(row)
                        elif row[0] and counter == 2: all_numCells.append(row)
                        elif row[0] and counter == 4: all_numShea.append(row)
                        elif row[0] and counter == 6: all_numMFLC.append(row)
                    elif not first and row:
                        if row[0] and counter == 0:   all_fibers[0].extend(row)
                        elif row[0] and counter == 2: all_numCells[0].extend(row)
                        elif row[0] and counter == 4: all_numShea[0].extend(row)
                        elif row[0] and counter == 6: all_numMFLC[0].extend(row)                                
                                    
                    if counter == 6:
                        break
                    counter = counter + 1
                
            if not empty:
                first = 0    
        #l = [all_fibers[0], all_numCells[0], all_numShea[0], all_numMFLC[0]]
            
        with open(output_name, 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            if all_fibers:
                wr.writerow(all_fibers[0])
                wr.writerow(all_numCells[0])
                wr.writerow(all_numShea[0])
                wr.writerow(all_numMFLC[0])    
            
        X = X + 1


""" Read and combine csv """
def read_and_comb_csv_FINAL_singles():
    all_fibers = []
    all_numCells = []
    all_numShea = []
    all_numMFLC = []
    
    fold_nam = 'uFNet-5_CSVs/3) combined_duplicates/'
    input_path = './' + fold_nam 
    all_csv = read_file_names(input_path)

    output_name = 'FINAL_ALL' + '_' + all_csv[0]    
    first = 1;
    with open(output_name, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for T in range(len(all_csv)):
            
            filename = all_csv[T]
            empty = 0
            with open(input_path + filename, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                counter = 0
            
                for row in spamreader:
                    print(', '.join(row))
                    
                    row = list(filter(lambda a: a != '[]', row))
                    
                    wr.writerow(row)


""" go through list_cells to get all the information """
def cycle_and_output_csv(list_cells, output_name, minLengthSingle, total_DAPI=0, total_matched_DAPI=0, s_path=''):
    num_wrap = 0
    wrap_per_cell = []
    all_fiber_lengths = []
    mFLC = []
    
    new_list = []
    for i in range(len(list_cells)):
        fibers = list_cells[i].fibers        
        if len(fibers) == 1 and fibers[0] < minLengthSingle:
            all_fiber_lengths.extend([]) 
            new_list.append([])
        elif len(fibers) == 2 and (fibers[0] < minLengthSingle and fibers[1] < minLengthSingle):
            all_fiber_lengths.extend([])
            new_list.append([])
        elif fibers:   # if it is NOT empty, then there are fibers
            num_wrap = num_wrap + 1
            wrap_per_cell.append(len(fibers))
            all_fiber_lengths.extend(fibers)
            mean = sum(fibers)/len(fibers)
            mFLC.append(mean)
            new_list.append(list_cells[i])
        else:
            all_fiber_lengths.extend([]) 
            new_list.append([])
    lis_props = [num_wrap]
    
    with open(s_path + output_name, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(all_fiber_lengths)
        wr.writerow(lis_props)
        wr.writerow(wrap_per_cell)
        wr.writerow(mFLC)
        wr.writerow([total_matched_DAPI])
        wr.writerow([total_DAPI])        
        
    return new_list
        
        
""" NEW: find all that have width too large """
def width_separate(masked, all_fibers, width_thresh, minLength):
    binary_all_fibers = masked > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    large_width = np.zeros(masked.shape)
    short_width = np.zeros(masked.shape)
    
    for M in range(len(cc_overlap)):
        length = cc_overlap[M]['MajorAxisLength']
        angle = cc_overlap[M]['Orientation']
        overlap_coords = cc_overlap[M]['coords']
        width = cc_overlap[M]['MinorAxisLength']
        
        if width > width_thresh and length > minLength and (angle > +0.785398 or angle < -0.785398):
            cell_num = cc_overlap[M]['MaxIntensity']
            cell_num = int(cell_num) 
    
            for T in range(len(overlap_coords)):
                large_width[overlap_coords[T,0], overlap_coords[T,1]] = cell_num

        else:
        
            for T in range(len(overlap_coords)):
                cell_num = cc_overlap[M]['MaxIntensity']
                cell_num = int(cell_num)      
            
                short_width[overlap_coords[T,0], overlap_coords[T,1]] = cell_num  
                
    """ CHANGED FROM 5,3 ==> 2,2 """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3))
    dil = cv2.dilate(large_width,kernel,iterations = 1)
    kernel = np.ones((8,1),np.uint8)    
    opening = cv2.morphologyEx(dil, cv2.MORPH_OPEN, kernel)
    combined = np.add(opening, short_width)
    
    return combined



def skel_one(all_fibers, minLength):
    image = all_fibers
    image = image > 0
    skeleton = skeletonize(image)
    
    bp = find_branch_points(skeleton)
    
    """ Then dilate the branchpoints and subtract from the original image """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bpd = cv2.dilate(bp,kernel,iterations = 1)
    bpd = bpd.astype(int)
    bpd[bpd > 0] = 1
    
    sub_im = skeleton - bpd
    sub_im[sub_im < 0] = 0
    sub_im[sub_im > 0] = 1
    
    """ Find EVERYTHING smaller than minLength, AND in wrong orientation, so can delete from whole image afterwards """
    smallLength = 0
    sub_im  = sub_im > 0
    labelled = measure.label(sub_im)
    cc_overlap = measure.regionprops(labelled)
    
    hor_lines = np.zeros(sub_im.shape)
    for i in range(len(cc_overlap)):
        length = cc_overlap[i]['MajorAxisLength']
        angle = cc_overlap[i]['Orientation']
        overlap_coords = cc_overlap[i]['coords']
        #print(angle)
        if length < smallLength or (angle <= +0.785398 and angle >= -0.785398):
    
            for T in range(len(overlap_coords)):
                hor_lines[overlap_coords[T,0], overlap_coords[T,1]] = 1
                    
    """ Then subtract the horizontal and too small lines from the original skeleton"""
    all_vert = skeleton - hor_lines                      
    
    """ then invert this, and use this as a mask over top of the "all_fibers" """    
    masked = all_fibers
    masked[all_vert == 0] = 0                

    """ SEPARATE BY WIDTH """
    width = 10
    combined = width_separate(masked, all_fibers, width, minLength)
    
    masked = all_fibers
    masked[combined == 0] = 0
    
    return masked
    
    
""" uses list_M_cells to find out where the DAPI nuclei are of cells with fibers """
def extract_ensheathed_DAPI(DAPI_tmp, list_cells):
    labelled = measure.label(DAPI_tmp)
    cc = measure.regionprops(labelled)

    DAPI_ensheathed = np.zeros(DAPI_tmp.shape)   
    num_cells = 0
    for i in range(len(list_cells)):
        fibers = list_cells[i].fibers
        if fibers and i < len(cc):
            overlap_coords = cc[i]['coords']
            for T in range(len(overlap_coords)):
                DAPI_ensheathed[overlap_coords[T,0], overlap_coords[T,1]] = i                
    
            num_cells = num_cells + 1
    
    return DAPI_ensheathed     


""" Takes an image and associates all the fibers in it to a list of cells"""
def fiber_to_list(masked, all_fibers, list_cells, minLength):
        
    binary_all_fibers = masked > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    final_counted = np.zeros(masked.shape)
    for M in range(len(cc_overlap)):
        length = cc_overlap[M]['MajorAxisLength']
        angle = cc_overlap[M]['Orientation']
        overlap_coords = cc_overlap[M]['coords']
   
        if length > minLength and (angle > +0.785398 or angle < -0.785398):
            cell_num = cc_overlap[M]['MaxIntensity']
            cell_num = int(cell_num) 

            list_cells[cell_num].add_fiber(length)
            list_cells[cell_num].add_coords(overlap_coords)          
            for T in range(len(overlap_coords)):
                final_counted[overlap_coords[T,0], overlap_coords[T,1]] = cell_num
                            
    return list_cells, final_counted


""" Take final list and turn it into an image """
def im_from_list(list_cells, minLengthSingle, shape):

    new_fibers = np.zeros(shape)  
    num_fibers = 0
    for i in range(len(list_cells)):
        if list_cells[i]:
            fibers =  list_cells[i].fibers 
            if len(fibers) == 1 and fibers[0] < minLengthSingle:
                continue;
            elif len(fibers) == 2 and (fibers[0] < minLengthSingle and fibers[1] < minLengthSingle):
                continue;
            elif fibers:   # if it is NOT empty, then there are fibers
                coords = list_cells[i].coords
                for T in range(len(coords)):
                    new_fibers[coords[T,0], coords[T,1]] = i
                    
                num_fibers = num_fibers + 1
                    
    return new_fibers


""" Skeletonize and output final cell count """
def skeletonize_all_fibers(all_fibers, i, DAPI_tmp, minLength, minLengthSingle, total_DAPI=0, total_matched_DAPI=0, s_path='', name='', jacc_test=0):

    im_num = i
    minLengthSingle = minLengthSingle
    #name = filename_split
    #s_path = sav_dir
    # Invert the image
    image = all_fibers
    image = image > 0
    
    """ different skeletonization methods to try out"""
    skeleton = skeletonize(image)        
    bp = find_branch_points(skeleton)
            
    """ Then dilate the branchpoints and subtract from the original image """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bpd = cv2.dilate(bp,kernel,iterations = 1)
    bpd = bpd.astype(int)
    bpd[bpd > 0] = 1
    
    sub_im = skeleton - bpd
    sub_im[sub_im < 0] = 0
    sub_im[sub_im > 0] = 1
    
    """ Find EVERYTHING smaller than minLength, AND in wrong orientation, so can delete from whole image afterwards """
    smallLength = 0
    sub_im  = sub_im > 0
    labelled = measure.label(sub_im)
    cc_overlap = measure.regionprops(labelled)
    
    hor_lines = np.zeros(sub_im.shape)
    for i in range(len(cc_overlap)):
        length = cc_overlap[i]['MajorAxisLength']
        angle = cc_overlap[i]['Orientation']
        overlap_coords = cc_overlap[i]['coords']
        #print(angle)
        if length < smallLength or (angle <= +0.785398 and angle >= -0.785398):
    
            for T in range(len(overlap_coords)):
                hor_lines[overlap_coords[T,0], overlap_coords[T,1]] = 1
                     
    
    """ Then subtract the horizontal and too small lines from the original skeleton"""
    all_vert = skeleton - hor_lines      
        
    """ then invert this, and use this as a mask over top of the "all_fibers" """
    masked = np.copy(all_fibers)
    masked[all_vert == 0] = 0

    """ Clean garbage """
    all_vert = [];bp = []; bpd = []; hor_lines = []; image = []; labelled = []; skeleton = []; sub_im = [];
        
    """ SEPARATE BY WIDTH """
    width = 10
    combined = width_separate(masked, all_fibers, width, minLength)

    masked = np.copy(masked)
    masked[combined == 0] = 0

    """ Eliminate anything smaller than minLength, and in wrong orientation, then add to cell object """
    num_MBP_pos = 8000
    N = 100000
        
    list_cells = []
    for M in range(N):
         cell = Cell(N)
         list_cells.append(cell)
    list_cells_sorted, final_counted = fiber_to_list(masked, all_fibers, list_cells, minLength)    

#    """ Subtract expanded ensheathed DAPI spots b/c no fibers can pass through cell nucleus """
#    DAPI_ensheathed = extract_ensheathed_DAPI(DAPI_tmp, list_cells)
#
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#    dilated_DAPI = cv2.dilate(DAPI_ensheathed,kernel,iterations = 1)
#
#    copy_final_counted = np.copy(final_counted)
#    copy_final_counted[dilated_DAPI > 0] = 0
#    
#    # clean GARBAGE
#    DAPI_ensheathed = []; dilated_DAPI = []; combined = [];
    
    list_cells = []
    for M in range(N):
         cell = Cell(N)
         list_cells.append(cell)
    list_cells_sorted, final_counted_new = fiber_to_list(final_counted, all_fibers, list_cells, minLength)

    """ go through list_cells to get all the information """
    output_name = 'masked_out_dil' + '_' + name + '_' + str(im_num) + '.csv'
    new_list = cycle_and_output_csv(list_cells_sorted, output_name, minLengthSingle, total_DAPI, total_matched_DAPI, s_path=s_path)

    shape = np.shape(DAPI_tmp)
    new_fibers = im_from_list(new_list, minLengthSingle, shape)
    #plt.imsave('final_image' + str(im_num) + '.tif', (new_fibers * 255).astype(np.uint16))
    
    sz = 5;
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz));   # get disk structuring element
    dil_final = cv2.dilate(new_fibers, kernel, 1)


    """ Print out pickles for jaccard testing """
    if jacc_test:   
        for_jaccard_testing(new_fibers, all_fibers, minLength, DAPI_tmp, im_num, N, s_path=s_path)

    
    return dil_final
    
    
    
    
    
    
""" FOR JACCARD TESTING """
def for_jaccard_testing(new_fibers, all_fibers, minLength, DAPI_tmp, im_num, N, s_path):
       
    import pickle as pickle
    """ Print text onto image """
    #output_name = 'masked_out_dil' + str(im_num) + '.png'
    #add_text_to_image(new_fibers, filename=output_name)
           
    """ Sort through the final DAPI ==> for Jaccard testing only"""
    list_cells = []
    for M in range(N):
         cell = Cell(N)
         list_cells.append(cell)
    list_cells_sorted, final_counted_new = fiber_to_list(new_fibers, all_fibers, list_cells, minLength)
    DAPI_ensheathed = extract_ensheathed_DAPI(DAPI_tmp, list_cells_sorted)
    plt.imsave(s_path + 'DAPI_ensheathed_second' + str(im_num) + '.tif', (DAPI_ensheathed * 255).astype(np.uint16))
    with open(s_path + 'DAPI_ensheathed' + str(im_num) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([DAPI_ensheathed], f)    
       
       
       
    """ Create for ==> GLOBAL JACCARD """
    # first find all unique values
    uniq = np.unique(new_fibers)
    
    binary_all_fibers = all_fibers > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    final_counted = np.zeros(all_fibers.shape)
    for Q in range(len(cc_overlap)):
        cell_num = cc_overlap[Q]['MinIntensity']
        cell_num = int(cell_num) 
        overlap_coords = cc_overlap[Q]['coords']
            
        fiber = 0
        for T in range(len(uniq)):
            if cell_num == uniq[T]:
                fiber = 1
                #print(uniq[T])
                break
    
        if fiber:
            for T in range(len(overlap_coords)):
                final_counted[overlap_coords[T,0], overlap_coords[T,1]] = cell_num

    import pickle
    with open(s_path + 'final_jacc_fibers' + str(im_num) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
       pickle.dump([final_counted], f)       
       
       
       
       