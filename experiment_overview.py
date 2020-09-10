import os
from datetime import datetime

import numpy as np
import pandas as pd
import data_processor as dtp

'''
path_de = 'raw_data/final_data/2278116_2020-08-23_12-44-44.txt'
path_am = 'raw_data/final_data/2287306_2020-08-23_12-45-28.txt'

data_de = pd.read_csv(path_de, sep='\t')
data_de.to_csv('wowoemotion_raw_de.csv', sep=';')

data_am = pd.read_csv(path_am, sep='\t')
data_am.to_csv('wowoemotion_raw_am.csv', sep=';')
'''

def create_experiment_overview(df_experiment, language):
    base_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    overview = pd.DataFrame(columns=['session', 'sex', 'age', 'language', 'foreign_languages'])

    sessions = df_experiment['sessionid'].unique()
    print(sessions, len(sessions))

    for session in sessions:
        session_data = df_experiment.loc[df_experiment['sessionid']==session]
        session_length = session_data.shape[0]
        if dtp.experiment_conditions_fulfilled(session_data, language):
            session_overview = pd.DataFrame(columns=['session', 'sex', 'age', 'language', 'foreign_languages'])
            session_overview['session'] = session_data['sessionid'].unique()
            session_overview['sex'] = session_data['sex'].unique()
            session_overview['age'] = session_data['participantage'].unique()
            session_overview['language'] = session_data['firstlanguage'].unique()
            session_overview['foreign_language'] = session_data['foreignlanguages'].unique()
            #itemis for the control question: armenian = 2287589, german = 2286153
            #control_q =  session_data.loc[(session_data['itemid']==2287589) | (session_data['itemid']==2286153)]
            #session_overview['control_question'] = control_q['inputvalue'].values[0] if control_q['inputvalue'].any() else 'Nothing'
            overview = pd.concat([overview, session_overview])
        else:
            print('Not complete session  wa found in normalized data: check for bugs!')




    overview.to_csv('overview/'+base_name + 'overview_' + language + '.csv', sep=';')





base_folder = 'normalized_data/'
file_list = dtp.get_filelist_in_folder(base_folder)

for f in file_list:
    f_path = base_folder + f
    annotation_data = pd.read_csv(f_path)
    experiment_language = dtp.get_experiment_language(annotation_data)
    create_experiment_overview(annotation_data, experiment_language)
