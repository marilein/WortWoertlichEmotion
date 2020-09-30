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

import json
import os
from datetime import datetime
import pandas as pd

# stimuli classes Original, Pitch and Tempo: mapping emotion abbreviations in file names with emotion labels
original_label_keys = ['_A.wav', '_E.wav', '_F.wav', '_s.wav', '_W.wav', '_T.wav']
original_label_values = ['Angst', 'Ekel', 'Freude', 'neutral', 'Wut', 'Trauer']
original_label_dict = dict(zip(original_label_keys, original_label_values))
pitch_label_keys = ['A_pitch', 'E_pitch', 'F_pitch', 's_pitch', 'W_pitch', 'T_pitch']
pitch_label_values = ['Angst_F0', 'Ekel_F0', 'Freude_F0', 'neutral_F0', 'Wut_F0', 'Trauer_F0']
pitch_label_dict = dict(zip(pitch_label_keys, pitch_label_values))
tempo_label_keys = ['A_tempo', 'E_tempo', 'F_tempo', 's_tempo', 'W_tempo', 'T_tempo']
tempo_label_values = ['Angst_tempo', 'Ekel_tempo', 'Freude_tempo', 'neutral_tempo', 'Wut_tempo', 'Trauer_tempo']
tempo_label_dict = dict(zip(tempo_label_keys, tempo_label_values))
all_labels_dict = {**original_label_dict, **pitch_label_dict, **tempo_label_dict}


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


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
            if file.endswith('.txt') or file.endswith('.csv'):
                file_list.append(file)
    return file_list


def read_raw_data(file):
    df = pd.read_csv(file, sep='\t')
    return df


def get_session_data(df, session_id):
    session_data = df.loc[df['sessionid'] == session_id]
    return session_data


def get_session_list(df):
    sessions = df['sessionid'].unique()
    return sessions


def get_question(dict):
    options = json.loads(dict)
    return options['label']


def check_session_length(df_session):
    if len(df_session) == 151:
        return True
    return False


def check_first_language(df_session, language):
    '''
    check for a valid first language according to the experiment language given as parameter
    '''
    language_options = []
    first_language = df_session['firstlanguage'].str.lower().unique()
    if 'de' in language.lower():
        language_options.extend(['deutsch', 'german'])
    else:
        language_options.extend(['arm', 'armenian', 'հայ', 'հայերեն', 'հայոց'])

    # if df_session['experiment'].str.lower().contains(language).any():
    for l in first_language:
        if l.strip() in language_options:
            return True
    return False


def check_participant_age(df_session):
    participant_age = df_session['participantage'].unique()
    if int(participant_age) < 90 and int(participant_age) > 10:
        return True
    return False


def check_foreign_languages(df_session):
    '''
    returns False if any of the foreign languages of the participant is invalid
    '''
    invalid_languages = ['deutsch', 'arm', 'armenian', 'հայ', 'հայերեն', 'հայոց']
    foreign_languages = df_session['foreignlanguages'].unique()

    for invalid_language in invalid_languages:
        if invalid_language in foreign_languages:
            return False
    return True


def get_data_with_condition(df, field, value):
    data = df.loc[df[field] == value]
    return data


def get_data_per_emotion(df, emotion):
    df = df[df['url'].notna()]
    df_emotion = df.loc[df['url'].str.contains(emotion)]
    return df_emotion


def check_control_question(df_session):
    '''
    control question was to choose 'applause' instead of an emotion, when they hear clappig hands instead of human speech
    '''
    # loc the row containing control question and check inputvalue
    control_question = df_session.loc[df_session['url'].str.contains('applause', na=False)]
    legal_inputvalue = df_session.loc[df_session['inputvalue'] == 'Klatschen']

    if (legal_inputvalue.shape[0] == 1 and
            control_question['inputvalue'].str.contains('Klatschen').any()):
        return True
    return False


def replace_armenian_labels(df):
    df_mapping = pd.read_csv('inputvalues_mapping.csv', sep='\t')
    label_mapping = dict(zip(df_mapping['inputvalues_am'], df_mapping['inputvalues_de']))
    df['inputvalue'].replace(label_mapping, inplace=True)
    return df


def get_experiment_language(df):
    experiment_language = 'am' if df['experiment'].str.contains('(AM)').any() else 'de'
    return experiment_language


def experiment_conditions_fulfilled(df_session, language):
    '''
    check if the experiment conditions are fulfilled for the given session
    1. 151 entries in 1 session
    2. age is valied (<90, >10)
    3. firstlanguage is the same as the experiment language
    4, foreignlanguage doesn't include german or armenian languages
    5  the answer for the control question is correct
    '''

    if check_session_length(df_session) and \
            check_participant_age(df_session) and \
            check_first_language(df_session, language) and \
            check_foreign_languages(df_session) and \
            check_control_question(df_session):
        return True

    return False


def fill_in_intended_emotions(df):
    pat = r'({})'.format('|'.join(all_labels_dict.keys()))
    extracted = df['url'].str.extract(pat, expand=False).dropna()
    df['expected'] = extracted.apply(lambda x: all_labels_dict[x]).reindex(df.index).fillna(0)
    return df


def extract_inputvalues(raw_path):
    annotaion_files = get_filelist_in_folder(raw_path)
    for f in annotaion_files:
        f_path = raw_path + f
        annotation_data = pd.read_csv(f_path, sep='\t')
        experiment_name = 'am' if annotation_data['experiment'].str.contains('(AM)').any() else 'de'
        inputvalues = annotation_data['inputvalue'].unique().tolist()

        df_inputvalues = pd.DataFrame(data={'inputvalues': inputvalues})
        df_inputvalues.to_csv(f'inputvalues_{experiment_name}.csv', sep=',', index=False)


def get_stimuli_annotations(df_annotations_all):
    df_stimuli_annotations = df_annotations_all[df_annotations_all['url'].notna()]
    df_emotion_annotations = df_stimuli_annotations.loc[~df_stimuli_annotations["url"].str.contains("applause")]
    return df_emotion_annotations
