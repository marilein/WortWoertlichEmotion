from _datetime import datetime
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import data_processor as dtp
import random

sessionid ='sessionid'
inputvalue = 'inputvalue'
url = 'url'



def answer_mode_per_task(df_aggregated, df, experiment):


    for stimulus in df_aggregated[url]:  # iterate over all available files and get max vote


        # compute majority answer only if there are at least 2 annotaions for the audio file
        if df.loc[df[url] == stimulus, inputvalue].shape[0] > 1:
            df_aggregated.loc[df_aggregated[url] == stimulus, experiment] = \
                df.loc[df[url] == stimulus, inputvalue].mode()[0]

    return df_aggregated


def compute_interrater_agreement(folder_path):
    annotaion_files = dtp.get_filelist_in_folder(folder_path)

    first = True
    # create an empty dataframe to fill with majority votes for both experiments
    df_aggregated = pd.DataFrame()

    for f in annotaion_files:

        f_path = folder_path + f
        df = pd.read_csv(f_path)
        df = df[~df[url].str.contains('applause')]
        experiment_name = 'am' if df['experiment'].str.contains('(AM)').any() else 'de'

        #create a dataframe for interrater agreement
        df_rater_agreement = pd.DataFrame(columns=[sessionid], data=df[sessionid].unique())

        #fill the stimulus aggregation dataframe with unique audiofiles only once (for the first experiment)
        if first:
            df_aggregated[url] = df[url].dropna().unique()
            df_aggregated = df_aggregated[~df_aggregated[url].str.contains('applause')]
            first=False

        df_aggregated = answer_mode_per_task(df_aggregated, df, experiment_name)


        for rater in df[sessionid].unique():  # iterate over all raters
            # get answers of that rater
            rater_annotations = df.loc[df[sessionid] == rater, [url, inputvalue]]
            rater_annotations = rater_annotations[url].dropna()
            # if there are any duplicates, remove all except the first one
            rater_annotations = rater_annotations.drop_duplicates(subset=url, keep="first")



            # get the average answers for the files this rater has annotated
            average_annotations = df_aggregated.loc[df_aggregated[url].isin(rater_annotations[url])]


            # sort both DataFrames by filename to make sure correlation is computed for the right pairs
            rater_annotations = rater_annotations.sort_values(by=url)
            average_annotations = average_annotations.sort_values(by=url)

            # compute accuracy score and store as rater weight
            # the name of the average answer column was 'AverageAnswer' - now chaging it to the question itself as a column name
            df_rater_agreement.loc[df_rater_agreement[sessionid] == rater, experiment_name] =\
                ("%.4f" % (accuracy_score(rater_annotations[inputvalue], average_annotations[experiment_name])))


        df_rater_agreement.to_csv(f'rater_agreement_{experiment_name}')

    df_aggregated.to_csv('majority_votes.csv')





base_folder = 'normalized_data/'
compute_interrater_agreement(base_folder)