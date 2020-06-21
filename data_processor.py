import os
from datetime import datetime


def get_datetime_str(dt):
    '''
    returns datetime in a standard format
    if datetime argument is not given, then return datetime now
    '''


    if dt:
        datetime_string = dt.strftime("%Y%m%d-%H%M%S")
    else:
        datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    return datetime_string


def get_filelist_in_folder(path):
    '''
    gets the files in the given folder and returns their path as a list
    '''
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                file_list.append(file)

    return file_list



