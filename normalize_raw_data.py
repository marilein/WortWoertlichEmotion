import IPython
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import data_processor as dtp

base_path = 'raw_data/final_data/'

#experiment_list = dtp.get_filelist_in_folder(base_path)

def create_plots_per_emotion(path):
    annotaion_files = dtp.get_filelist_in_folder(path)

    for f in annotaion_files:
        f_path = path + '/' + f
        annotation_data = dtp.read_raw_data(f_path)
        experiment_name = 'am' if annotation_data['experiment'].str.contains('(AM)').any() else 'de'
        #json_question = annotation_data['options'].apply(json.loads)

        session_list = dtp.get_session_list(annotation_data)
        experiment_plots = 'plots/'+experiment_name
        if not os.exists(experiment_plots):
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)


        '''
        for session in session_list:
            #df_session = annotation_data.loc[annotation_data['sessionid']==session]
            if dtp.check_session_length(session):
                df_session = dtp.get_session_data(annotation_data, session)

            #TODO check whether the reuquirements are fulfilled
            pass
        '''


