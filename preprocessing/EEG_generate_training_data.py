#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os, sys
import numpy as np
# from EEG_feature_extraction import generate_feature_vectors_from_samples


def gen_training_matrix(directory_path, output_file, cols_to_ignore):
    """
    Reads the csv files in directory_path and assembles the training matrix with 
    the features extracted using the functions from EEG_feature_extraction.
    
    Parameters:
        directory_path (str): directory containing the CSV files to process.
        output_file (str): filename for the output file.
        cols_to_ignore (list): list of columns to ignore from the CSV

    Returns:
        numpy.ndarray: 2D matrix containing the data read from the CSV
    
    """
    
    # Initialise return matrix
    FINAL_MATRIX = None
    
    for x in os.listdir(directory_path):

        # Ignore non-CSV files
        if not x.lower().endswith('.csv'):
            continue
        
        # For safety we'll ignore files containing the substring "test". 
        # [Test files should not be in the dataset directory in the first place]
        if 'test' in x.lower():
            continue
        try:
            name, state, _ = x[:-4].split('-')
        except:
            print('Wrong file name', x)
            sys.exit(-1)
        if state.lower() == 'positive':
            state = 2.
        elif state.lower() == 'neutral':
            state = 1.
        elif state.lower() == 'negative':
            state = 0.
        else:
            print('Wrong file name', x)
            sys.exit(-1)
            
        print('Using file', x)
        full_file_path = directory_path + '/' + x
        vectors, header = generate_feature_vectors_from_samples(file_path=full_file_path, 
                                                                nsamples=150, 
                                                                period=1.,
                                                                state=state,
                                                                remove_redundant=True,
                                                                cols_to_ignore=cols_to_ignore)
        
        print('Resulting vector shape for the file', vectors.shape)
        
        if FINAL_MATRIX is None:
            FINAL_MATRIX = vectors
        else:
            FINAL_MATRIX = np.vstack([FINAL_MATRIX, vectors])

    print('FINAL_MATRIX', FINAL_MATRIX.shape)
    
    # Shuffle rows
    np.random.shuffle(FINAL_MATRIX)
    
    # Save to file
    np.savetxt(output_file, FINAL_MATRIX, delimiter=',',
               header=','.join(header), 
               comments='')

    return None


if __name__ == '__main__':

    # Hardcoded parameters
    directory_path = r"C:\MY Laptop\Emotion Detection\arduino" 
    output_file = r"C:\MY Laptop\Emotion Detection\data.csv"  
    cols_to_ignore = None 

    gen_training_matrix(directory_path, output_file, cols_to_ignore)
