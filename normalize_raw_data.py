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

import os
from re import search
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import data_processor as dtp

sns.set(color_codes=True)


def create_plots_per_emotion(path):
    annotaion_files = dtp.get_filelist_in_folder(path)

    for f in annotaion_files:
        f_path = path + '/' + f
        annotation_data = pd.read_csv(f_path)
        experiment_name = 'am' if annotation_data['experiment'].str.contains('(AM)').any() else 'de'
        # json_question = annotation_data['options'].apply(json.loads)

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
            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            plt.title(experiment_name + '_' + original_label)
            plt.show()

        for pitch_label in pitch_label_list:
            label_data = dtp.get_data_per_emotion(annotation_data, pitch_label)
            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            plt.title(experiment_name + '_' + pitch_label)
            plt.show()

        for tempo_label in tempo_label_list:
            label_data = dtp.get_data_per_emotion(annotation_data, tempo_label)
            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            plt.title(experiment_name + '_' + tempo_label)
            plt.show()


def normalize_raw_data(base_path):
    raw_files = dtp.get_filelist_in_folder(base_path)

    for f in raw_files:
        f_path = base_path + f
        annotation_data = pd.read_csv(f_path, sep='\t')
        experiment_language = 'am' if annotation_data['experiment'].str.contains('(AM)').any() else 'de'
        # json_question = annotation_data['options'].apply(json.loads)

        if experiment_language == 'am':
            annotation_data = dtp.replace_armenian_labels(annotation_data)
            annotation_data.to_csv('wowoemotion_mapped_am.csv')

        df_normalized = pd.DataFrame(columns=annotation_data.columns)
        session_list = dtp.get_session_list(annotation_data)
        inputvalue_klatschen = pd.DataFrame(columns=['sessionid', 'Klatschen', 'Eingaben'])
        inputvalue_klatschen['sessionid'] = annotation_data['sessionid'].unique()
        for session_id in session_list:
            df_session = dtp.get_session_data(annotation_data, session_id)
            legal_inputvalue = df_session.loc[df_session['inputvalue'] == 'Klatschen']
            # inputvalue_klatschen['sessionid'] = session_id
            inputvalue_klatschen.loc[inputvalue_klatschen['sessionid'] == session_id, 'Klatschen'] = \
            legal_inputvalue.shape[0]
            inputvalue_klatschen.loc[inputvalue_klatschen['sessionid'] == session_id, 'Eingaben'] = \
                df_session.shape[0]
            if dtp.experiment_conditions_fulfilled(df_session, experiment_language):
                df_normalized = pd.concat([df_normalized, df_session], ignore_index=True)

        # following block has been moved to data_processor module
        '''
        pat = r'({})'.format('|'.join(dtp.all_labels_dict.keys()))
        extracted = df_normalized['url'].str.extract(pat, expand=False).dropna()

        df_normalized['expected'] = extracted.apply(lambda x: dtp.all_labels_dict[x].split('_')[0]).reindex(df_normalized.index).fillna(0)
        '''
        df_normalized = dtp.fill_in_intended_emotions(df_normalized)
        df_naturaleness = df_normalized.loc[df_normalized['itemid'].isin([2278419, 2287438])]
        df_naturaleness['inputvalue'].to_csv(f'naturalness_{experiment_language}.csv', index=False)
        inputvalue_klatschen.to_csv(f'overview/participants/overview/inputvalue_Klatschen_{experiment_language}.csv',
                                    index=False)
        df_normalized.to_csv(f'normalized_data/normalized_final_{experiment_language}.csv', index=False)


def create_participation_overview(participants_folder):
    files = dtp.get_filelist_in_folder(participants_folder)
    df_overview = pd.DataFrame(columns=['Experiment', 'Teilnahmen', 'Done', 'Bestanden'])
    df_overview['Experiment'] = ['Deutsch', 'Armenisch']
    for f in files:
        f_path = participants_folder + f
        experiment_overview = pd.read_csv(f_path)
        experiment_language = 'Armenisch' if search('_am', f) else 'Deutsch'
        df_overview.loc[df_overview['Experiment'] == experiment_language, ['Teilnahmen', 'Done', 'Bestanden']] = \
            [experiment_overview.shape[0], experiment_overview.loc[experiment_overview['Eingaben'] == 151].shape[0],
             'add']

    df_overview.to_csv('overview/participants/analysis/participation_overview.csv', index=False)


def merge_experiments_data(data_folder):
    annotation_files = dtp.get_filelist_in_folder(data_folder)
    df_all = pd.DataFrame()

    for annotation_file in annotation_files:
        f_path = data_folder + annotation_file
        df_data = pd.read_csv(f_path)
        df_all = pd.concat([df_all, df_data], ignore_index=True)

    df_filtered = dtp.get_stimuli_annotations(df_all)

    df_filtered.to_csv('analyze/normalized_emotion_annotations_all.csv')


if __name__ == "__main__":
    raw_path = 'raw_data/final_data/'
    normalize_raw_data(raw_path)

    normalized_path = './normalized_data'
    participants_folder = 'overview/participants/overview/'
    create_plots_per_emotion(normalized_path)
    merge_experiments_data('normalized_data/')
    create_participation_overview(participants_folder)
