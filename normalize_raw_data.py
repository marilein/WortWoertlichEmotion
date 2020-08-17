import IPython
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import data_processor as dtp
sns.set(color_codes=True)


base_path = './normalized_data'

#experiment_list = dtp.get_filelist_in_folder(base_path)

def create_plots_per_emotion(path):
    annotaion_files = dtp.get_filelist_in_folder(path)

    for f in annotaion_files:
        f_path = path + '/' + f
        annotation_data = pd.read_csv(f_path)
        experiment_name = 'am' if annotation_data['experiment'].str.contains('(AM)').any() else 'de'
        #json_question = annotation_data['options'].apply(json.loads)

        original_label_list = ['_A.wav', '_E.wav', '_F.wav', '_s.wav', '_W.wav', '_T.wav']
        pitch_label_list = ['A_pitch', 'E_pitch', 'F_pitch', 's_pitch', 'W_pitch', 'T_pitch']
        tempo_label_list = ['A_tempo', 'E_tempo', 'F_tempo', 's_tempo', 'W_tempo', 'T_tempo']

        experiment_plots = 'plots/' + experiment_name
        if not os.path.exists(experiment_plots):
            try:
                os.makedirs(experiment_plots)
            except OSError:
                print("Creation of the directory %s failed" % path)

        for original_label in original_label_list:
            label_data = dtp.get_data_per_emotion(annotation_data, original_label)
            #ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            #ax = sns.barplot(x="inputvalue", y="x", data=label_data, estimator=lambda x: len(x) / label_data.shape[0] * 100)
            ax = sns.countplot(x="inputvalue", data=label_data, order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            plt.title(experiment_name + '_' + original_label)
            plt.show()

        for pitch_label in pitch_label_list:
            label_data = dtp.get_data_per_emotion(annotation_data, pitch_label)
            #ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            #ax = sns.barplot(x="inputvalue", y="inputvalue", data=label_data,estimator=lambda x: len(x) / label_data.shape[0] * 100)
            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            plt.title(experiment_name + '_' + pitch_label)
            plt.show()

        for tempo_label in tempo_label_list:
            label_data = dtp.get_data_per_emotion(annotation_data, tempo_label)
            #ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            #ax = sns.barplot(x="inputvalue", y="inputvalue", data=label_data,estimator=lambda x: len(x) / label_data.shape[0] * 100)
            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            plt.title(experiment_name + '_' + tempo_label)
            plt.show()

        session_list = dtp.get_session_list(annotation_data)



        '''
        for session in session_list:
            #df_session = annotation_data.loc[annotation_data['sessionid']==session]
            if dtp.check_session_length(session):
                df_session = dtp.get_session_data(annotation_data, session)

            #TODO check whether the reuquirements are fulfilled
            pass
        '''


create_plots_per_emotion(base_path)