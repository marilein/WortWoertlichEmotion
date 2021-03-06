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

from re import search
import pandas as pd
import data_processor as dtp
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from scipy.stats import chi2_contingency


def calculate_percentage_of_right_answers():
    normalized_path = './normalized_data/'
    files = dtp.get_filelist_in_folder(normalized_path)
    # some frequent variables
    emotion = 'Emotion'
    lang = 'Language'
    original = 'Original [%]'
    f0 = 'F0 [%]'
    tempo = 'Tempo [%]'

    df_right_answers_all = pd.DataFrame(columns=[lang, emotion, original, f0, tempo])

    for f in files:
        f_path = normalized_path + f
        experiment_data = pd.read_csv(f_path)
        experiment_language = 'Armenisch' if search('_am', f) else 'Deutsch'
        # print(experiment_language, '\n', experiment_data.describe())

        df_experiment_right_answers = pd.DataFrame(columns=[lang, emotion, original, f0, tempo])
        df_experiment_right_answers[emotion] = ['Freude', 'Ekel', 'neutral', 'Trauer', 'Angst', 'Wut']

        for original_label in dtp.original_label_keys:
            label_key = original_label
            label_value = dtp.original_label_dict[label_key]
            label_data = dtp.get_data_per_emotion(experiment_data, label_key)
            annot_count = label_data.shape[0]
            label_count = label_data.loc[label_data['inputvalue'] == label_value].shape[0]
            ratio = label_count * 100 / annot_count
            df_experiment_right_answers.loc[
                df_experiment_right_answers[emotion] == label_value, lang] = experiment_language
            df_experiment_right_answers.loc[
                df_experiment_right_answers[emotion] == label_value, original] = '{:.1f}'.format(ratio)

        for f0_label in dtp.pitch_label_keys:
            label_key = f0_label
            label_value = dtp.pitch_label_dict[label_key].split('_')[0]
            label_data = dtp.get_data_per_emotion(experiment_data, label_key)
            annot_count = label_data.shape[0]
            label_count = label_data.loc[label_data['inputvalue'] == label_value].shape[0]
            ratio = label_count * 100 / annot_count
            df_experiment_right_answers.loc[
                df_experiment_right_answers[emotion] == label_value, lang] = experiment_language
            df_experiment_right_answers.loc[df_experiment_right_answers[emotion] == label_value, f0] = '{:.1f}'.format(
                ratio)

        for tempo_label in dtp.tempo_label_keys:
            label_key = tempo_label
            label_value = dtp.tempo_label_dict[label_key].split('_')[0]
            label_data = dtp.get_data_per_emotion(experiment_data, label_key)
            annot_count = label_data.shape[0]
            label_count = label_data.loc[label_data['inputvalue'] == label_value].shape[0]
            ratio = label_count * 100 / annot_count
            df_experiment_right_answers.loc[
                df_experiment_right_answers[emotion] == label_value, lang] = experiment_language
            df_experiment_right_answers.loc[
                df_experiment_right_answers[emotion] == label_value, tempo] = '{:.1f}'.format(ratio)

        df_right_answers_all = pd.concat([df_right_answers_all, df_experiment_right_answers], ignore_index=True)

    df_right_answers_all.to_csv('right_answers_percentage_all.csv', index=False)


def get_language_and_group(f_path):
    lang = f_path.split('_')[2]
    language = 'Armenisch' if lang == 'am' else 'Deutsch'
    tail = f_path.split('_')[3]
    group = tail.split('.')[0]

    return language, group


def calculate_percentage_of_all_answers():
    conf_folder = 'confusion_matrices/'
    f_list = dtp.get_filelist_in_folder(conf_folder)
    all_columns = ['Language', 'Group', 'Emotion'] + dtp.original_label_values
    df_conf_percent_all = pd.DataFrame(columns=all_columns)

    for f in f_list:
        f_path = conf_folder + f

        language, group = get_language_and_group(f)
        df = pd.read_csv(f_path)
        df_conf_percent = pd.DataFrame(columns=all_columns)
        df_conf_percent['Emotion'] = dtp.original_label_values
        annot_count = df[dtp.original_label_values].sum(axis=1).unique()

        for true_label in dtp.original_label_values:
            df_conf_percent.loc[df_conf_percent['Emotion'] == true_label, 'Language'] = language
            df_conf_percent.loc[df_conf_percent['Emotion'] == true_label, 'Group'] = group

            for pred_label in dtp.original_label_values:
                pred_label_count = df.loc[df['Prosodischer Ausdruck'] == true_label, pred_label]
                ratio = 100 * pred_label_count / annot_count
                df_conf_percent.loc[df_conf_percent['Emotion'] == true_label, pred_label] = "%.1f" % ratio

        df_conf_percent_all = pd.concat([df_conf_percent_all, df_conf_percent], ignore_index=True)

    df_conf_percent_all.to_csv('confusion_matrices_percentage_all.csv', index=False)


def calculate_anotation_percentage():
    calculate_percentage_of_right_answers()
    calculate_percentage_of_all_answers()


def count_answers_for_experiment(df_obs, df_label):
    language = 'Armenisch' if df_label['experiment'].str.contains('(AM)').any() else 'Deutsch'
    for emotion in dtp.original_label_values:
        df_obs.loc[df_obs['Language'] == language, emotion] = df_label.loc[df_label['inputvalue'] == emotion].shape[0]
    return df_obs


def get_label_data_for_chi2(df_label):
    df_chi_obs = pd.DataFrame(columns=['Language', 'Emotion'])
    df_temp = df_label[['inputvalue', 'experiment']]
    df_temp['Language'] = ['Armenisch' if search('AM', x) else 'Deutsch' for x in df_temp['experiment'].values]
    df_chi_obs[['Language', 'Emotion']] = df_temp[['Language', 'inputvalue']]
    return df_chi_obs


def compute_culture_contingency(df):
    subsets = ['original', 'pitch', 'tempo']
    groups = ['Armenisch', 'Deutsch']

    df_results = pd.DataFrame(columns=['Variables', 'xsq', 'pvalue', 'dof'])
    for subset in subsets:
        label_keys = f'dtp.{subset}_label_keys'
        df_subset = df.loc[df['url'].str.contains('|'.join(eval(label_keys)))]
        for label_key in getattr(dtp, f'{subset}_label_keys'):
            label_dict = getattr(dtp, f'{subset}_label_dict')
            label_value = label_dict[label_key]
            df_label = dtp.get_data_per_emotion(df_subset, label_key)
            # create observation df
            all_columns = ['Language'] + dtp.original_label_values
            df_obs = pd.DataFrame(columns=all_columns)
            df_obs['Language'] = groups

            for experiment in df_label['experiment'].unique():
                df_experiment = df_label.loc[df_label['experiment'] == experiment]
                df_obs = count_answers_for_experiment(df_obs, df_experiment)

            df_obs.to_csv(f'analyze/{subset}_{label_value}_counts.csv', index=False)
            obs_list = df_obs[dtp.original_label_values].to_numpy().tolist()
            df_for_chi = get_label_data_for_chi2(df_label)
            xsq, pvalue, dof, expected = chi2_contingency(obs_list)
            test_case = f'{subset}_{label_value}'
            df_results_temp = pd.DataFrame(columns=['Variables', 'xsq', 'pvalue', 'dof'])
            df_results_temp['Variables'] = [test_case]
            df_results_temp['xsq'] = xsq
            df_results_temp.loc[df_results_temp['Variables'] == test_case, 'pvalue'] = pvalue
            df_results_temp.loc[df_results_temp['Variables'] == test_case, 'dof'] = dof
            df_results = pd.concat([df_results, df_results_temp], ignore_index=True)

            # print(f'{subset} {label_value}: xsq: {xsq}, pvalue: {pvalue}, dof: {dof}, expected: {expected}')
            df_for_chi.to_csv(f'analyze/{subset}_{label_value}_for_chi.csv', index=False)
            df_label.to_csv(f'analyze/{subset}_{label_value}.csv', index=False)

    df_results.to_csv('results/chi2_results_between_cultures.csv', index=False)


def count_answers_original_and_subset(df_obs, df_subset, subset):
    for emotion in dtp.original_label_values:
        df_obs.loc[df_obs['Subset'] == subset, emotion] = df_subset.loc[df_subset['inputvalue'] == emotion].shape[0]
    return df_obs


def compute_manipulation_contingency(df):
    subsets = ['pitch', 'tempo']
    groups = ['Armenisch', 'Deutsch']

    # fill 'expected' column with intended emotion labels
    df = dtp.fill_in_intended_emotions(df)
    df_results = pd.DataFrame(columns=['Variables', 'xsq', 'pvalue', 'dof'])
    df_fischers = pd.DataFrame(columns=['Variables', 'oddsratio', 'pvalue'])
    for group in groups:
        language = 'DE' if group == groups[1] else 'AM'
        df_group = df.loc[df['experiment'].str.contains(language)]

        for subset in subsets:
            subset_values = dtp.original_label_values + getattr(dtp, f'{subset}_label_values')
            df_subset = df_group[df_group['expected'].isin(subset_values)]
            # create observation df
            all_columns = ['Subset'] + dtp.original_label_values
            df_obs = pd.DataFrame(columns=all_columns)
            df_obs['Subset'] = df_subset['expected'].unique()

            for subset_label in df_subset['expected'].unique():
                df_subset_label = df_subset.loc[df_subset['expected'] == subset_label]
                df_obs = count_answers_original_and_subset(df_obs, df_subset_label, subset_label)

            df_obs.to_csv(f'analyze/{group}_{subset}_counts.csv', index=False)

            # run chi2 for each emotion between original and manipulated
            for emotion in dtp.original_label_values:
                df_obs_emotion = df_obs[df_obs['Subset'].str.contains(emotion)]
                obs_list = df_obs_emotion[dtp.original_label_values].to_numpy().tolist()

                expected = stats.contingency.expected_freq(obs_list)
                if not 0 in expected:
                    xsq, pvalue, dof, expected = chi2_contingency(obs_list, correction=True)
                    test_case = f'{group}_{subset}_{emotion}'
                    df_results_temp = pd.DataFrame(columns=['Variables', 'xsq', 'pvalue', 'dof'])
                    df_results_temp['Variables'] = [test_case]
                    df_results_temp['xsq'] = xsq
                    df_results_temp.loc[df_results_temp['Variables'] == test_case, 'pvalue'] = pvalue
                    df_results_temp.loc[df_results_temp['Variables'] == test_case, 'dof'] = dof
                    df_results = pd.concat([df_results, df_results_temp], ignore_index=True)
                    '''     
                    print(
                        f'{group} {subset} {emotion}: xsq: {xsq}, pvalue: {pvalue}, dof: {dof}, expected: {expected}')
                    '''

                else:
                    obs_fischer = pd.DataFrame(columns=['Subset', 'right', 'wrong'])
                    obs_fischer['Subset'] = df_obs_emotion['Subset'].unique()
                    columns = []
                    columns.extend([x for x in dtp.original_label_values if x != emotion])

                    for sub in df_obs_emotion['Subset'].unique():
                        obs_fischer.loc[obs_fischer['Subset'] == sub, 'right'] = \
                            df_obs_emotion.loc[df_obs_emotion['Subset'] == sub, emotion].iloc[0]
                        df_temp = df_obs_emotion[df_obs_emotion['Subset'] == sub]
                        wrong = df_temp[columns].values.sum()
                        obs_fischer.loc[obs_fischer['Subset'] == sub, 'wrong'] = wrong

                    obs_fischer_list = obs_fischer[['right', 'wrong']].to_numpy().tolist()
                    oddsratio, pvalue = stats.fisher_exact(obs_fischer_list)
                    test_case = f'{group}_{subset}_{emotion}'
                    df_fischers_temp = pd.DataFrame(columns=df_fischers.columns)
                    df_fischers_temp['Variables'] = [test_case]
                    df_fischers_temp.loc[df_fischers_temp['Variables'] == test_case, 'oddsratio'] = oddsratio
                    df_fischers_temp.loc[df_fischers_temp['Variables'] == test_case, 'pvalue'] = pvalue
                    df_fischers = pd.concat([df_fischers, df_fischers_temp], ignore_index=True)

    df_results.to_csv('results/chi2_results_manipulation.csv', index=False)
    df_fischers.to_csv('results/fischers_results_manipulation.csv', index=False)


if __name__ == "__main__":
    df = pd.read_csv('analyze/normalized_emotion_annotations_all.csv')
    compute_culture_contingency(df)
    compute_manipulation_contingency(df)
