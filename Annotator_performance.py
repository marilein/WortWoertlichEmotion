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

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
import data_processor as dtp

sessionid = 'sessionid'
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
        df = df[df[url].str.contains('applause') == False]
        experiment_name = 'am' if df['experiment'].str.contains('(AM)').any() else 'de'
        # create a dataframe for interrater agreement
        df_rater_agreement = pd.DataFrame(columns=[sessionid], data=df[sessionid].unique())
        # fill the stimulus aggregation dataframe with unique audiofiles only once (for the first experiment)
        if first:
            df_aggregated[url] = df[url].dropna().unique()
            df_aggregated = df_aggregated[df_aggregated[url].str.contains('applause') == False]
            first = False

        df_aggregated = answer_mode_per_task(df_aggregated, df, experiment_name)

        for rater in df[sessionid].unique():  # iterate over all raters
            # get answers of that rater
            rater_annotations = df.loc[df[sessionid] == rater, [url, inputvalue]]
            rater_annotations[url].dropna()
            # if there are any duplicates, remove all except the first one
            rater_annotations = rater_annotations.drop_duplicates(url, keep='first')
            # get the average answers for the files this rater has annotated
            average_annotations = df_aggregated.loc[df_aggregated[url].isin(rater_annotations[url])].dropna()
            # sort both DataFrames by filename to make sure correlation is computed for the right pairs
            rater_annotations = rater_annotations.sort_values(by=url)
            average_annotations = average_annotations.sort_values(by=url)
            # compute accuracy score and store as rater weight
            # the name of the average answer column was 'AverageAnswer' - now chaging it to the question itself as a column name
            df_rater_agreement.loc[df_rater_agreement[sessionid] == rater, experiment_name] = \
                ("%.2f" % (accuracy_score(rater_annotations[inputvalue], average_annotations[experiment_name])))

        df_rater_agreement.to_csv(f'rater_agreement_{experiment_name}')

    df_aggregated.to_csv('majority_votes.csv')


def plot_rater_agreement():
    rater_agreement_am = 'rater_agreement_am.csv'
    rater_agreement_de = 'rater_agreement_de.csv'
    df_am = pd.read_csv(rater_agreement_am)
    df_de = pd.read_csv(rater_agreement_de)

    # plot ewe (evaluator weighted estimator) values for armenian participants
    a = sns.regplot(x='Rater', y='EWE', data=df_am, fit_reg=False)
    plt.xlabel('Teilnehmer')
    plt.ylabel('r')
    plt.ylim(0, 1)
    plt.title('Interrater-Reliabilität der armenischen Teilnehmer')
    plt.setp(a.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig('ewe_am.png')
    plt.close()

    # plot ewe (evaluator weighted estimator) values for german participants
    d = sns.regplot(x='Rater', y='EWE', data=df_de, fit_reg=False)
    plt.xlabel('Teilnehmer')
    plt.ylabel('r')
    plt.ylim(0, 1)
    plt.title('Interrater-Reliabilität der deutschen Teilnehmer')
    plt.setp(d.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig('ewe_de.png')
    plt.close()


if __name__ == "__main__":
    base_folder = 'normalized_data/'
    compute_interrater_agreement(base_folder)
    plot_rater_agreement()
