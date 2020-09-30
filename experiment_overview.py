"""
 ************************************************************************
 * This file is part of the Master thesis of the author.                *
 *                                                                      *
 * Project: "Wortwörtlich Emotion" - a web-experiment for studying      *
 * cross-cultural emotion perception                                    *
 *                                                                      *
 *  @author: Mariam Hemmer                                              *
 ************************************************************************
"""


import numpy as np
import pandas as pd
from datetime import datetime
import data_processor as dtp


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
            overview = pd.concat([overview, session_overview])
        else:
            print('Not complete session  wa found in normalized data: check for bugs!')

    overview.to_csv('overview/'+base_name + 'overview_' + language + '.csv', sep=';')



def participants_overview():
    overview_folder = 'overview/final/'
    file_list = dtp.get_filelist_in_folder(overview_folder)
    df_participants = pd.DataFrame(columns=['Kultur', 'count_M', 'count_F', 'count_all','mean_age_all', 'min_age_all',
                                            'max_age_all', 'SD_age_all','M_mean_age', 'M_SD_age','F_mean_age', 'F_SD_age'])
    df_participants['Kultur'] = ['Deutsch', 'Armenisch']
    for f in file_list:
        f_path = overview_folder + f
        experiment_overview = pd.read_csv(f_path, sep=';')
        experiment_language = 'Deutsch' if experiment_overview['language'].str.contains('de').any() else 'Armenisch'
        df_m = experiment_overview.loc[experiment_overview['sex'] == 'M']
        df_f = experiment_overview.loc[experiment_overview['sex'] == 'F']
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'count_M'] = df_m.shape[0]
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'count_F'] = df_f.shape[0]
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'count_all'] = experiment_overview.shape[0]
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'mean_age_all'] = experiment_overview['age'].mean()
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'min_age_all'] = experiment_overview['age'].min()
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'max_age_all'] = experiment_overview[
            'age'].max()
        df_participants.loc[df_participants['Kultur'] == experiment_language,  'SD_age_all'] = np.std(experiment_overview['age'].to_list())
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'M_mean_age'] = df_m['age'].mean()
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'M_SD_age'] = np.std(df_m['age'].to_list())
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'F_mean_age'] = df_f['age'].mean()
        df_participants.loc[df_participants['Kultur'] == experiment_language, 'F_SD_age'] = np.std(df_f['age'].to_list())

    df_participants.to_csv('participants_overview.csv')


def experiment_overview_for_folder():

    base_folder = 'normalized_data/'
    file_list = dtp.get_filelist_in_folder(base_folder)
    for f in file_list:
        f_path = base_folder + f
        annotation_data = pd.read_csv(f_path)
        experiment_language = dtp.get_experiment_language(annotation_data)
        create_experiment_overview(annotation_data, experiment_language)


if __name__=="__main__":

    experiment_overview_for_folder()
    participants_overview()

