import os
import numpy as np
import pandas as pd
import datetime
import json

path = './raw_data/final_data'

#some variables
experiment_id_de = '2278116'
experiment_id_am = '2287306'

def get_annotation_files(path):
    experiment_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                experiment_list.append(file)

    return experiment_list


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

def big_five_mapping(language):
    #TODO 1: create a mapping list for both experiments
    #TODO 2: append values to dict

    openness  = '2287443' if language == 'am' else '2278270'
    contiousness  = '2287441' if language == 'am' else '2278271'
    extraversion  = '2287440' if language == 'am' else '2278272'
    agreeableness  = '2287444' if language == 'am' else '2278273'
    neuroticism  = '2287442' if language == 'am' else '2278274'

    pass



column_list = ['itemid', 'url', 'experiment', 'options']

def create_itemlist_per_experiment(path):
    annotaion_files = get_annotation_files(path)

    for f in annotaion_files:
        f_path = path + '/' + f
        annotation_data = read_raw_data(f_path)
        experiment_name = 'am' if annotation_data['experiment'].str.contains('(AM)').any() else 'de'
        #json_question = annotation_data['options'].apply(json.loads)
        df_items = annotation_data[column_list].drop_duplicates()
        df_items['options'] = df_items['options'].apply(lambda x: get_question(x))
        df_items.to_csv('./item_mapping/itemlist_'+experiment_name+'.csv', sep=';')


create_itemlist_per_experiment(path)

