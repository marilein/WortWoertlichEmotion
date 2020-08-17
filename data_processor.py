import json
import os
from datetime import datetime
import pandas as pd

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

    if df_session['experiment'].str.lower().contains(language).any():
        return True

    return False


def check_participant_age (df_session):

    participant_age = df_session['participantage'].unique()
    if  int(participant_age) < 90 and int(participant_age) > 10:
        return True

    return False

def check_foreign_languages(df_session):
    '''
    returns False if any of the foreign languages of the participant is invalid
    '''
    invalid_languages = ['deutsch', 'arm', 'armenian', 'հայ', 'հայերեն', 'հայոց']
    foreign_languages = df_session['foreignlanguage'].unique()

    for invalid_language in invalid_languages:
        if invalid_language.isin(foreign_languages):
            return False

    return True


def get_data_with_condition(df, field, value):

    data = df.loc[df[field]==value]

    return data

def get_data_per_emotion(df, emotion):
    df = df[df['url'].notna()]
    df_emotion = df.loc[df['url'].str.contains(emotion)]


    return df_emotion

def check_control_question(df_session):
    '''
    control question was to choose 'applause' instead of an emotion, when they hear clappig hands instead of human speech
    '''

    #loc the row containing control question and check inputvalue

    pass


def experiment_conditions_fulfilled(df_session, language):
    '''
    check if the experiment conditions are fulfilled for the given session
    1. 151 entries in 1 session
    2. age is valied (<90, >10)
    3. firstlanguage is the same as the experiment language
    4, foreignlanguage doesn't include german or armenian languages
    5  the answer for the control question is correct
    '''

    pass

