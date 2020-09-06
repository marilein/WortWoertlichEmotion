import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_processor as dtp



original_label_keys = ['_A.wav', '_E.wav', '_F.wav', '_s.wav', '_W.wav', '_T.wav']
original_label_values = ['Angst', 'Ekel', 'Freude', 'neutral', 'Wut', 'Trauer']
original_label_dict = dict(zip(original_label_keys, original_label_values))
pitch_label_keys  = ['A_pitch', 'E_pitch', 'F_pitch', 's_pitch', 'W_pitch', 'T_pitch']
pitch_label_values = ['Angst_F0', 'Ekel_F0', 'Freude_F0', 'neutral_F0', 'Wut_F0', 'Trauer_F0']
pitch_label_dict = dict(zip(pitch_label_keys, pitch_label_values))
tempo_label_keys  = ['A_tempo', 'E_tempo', 'F_tempo', 's_tempo', 'W_tempo', 'T_tempo']
tempo_label_values = ['Angst_tempo', 'Ekel_tempo', 'Freude_tempo', 'neutral_tempo', 'Wut_tempo', 'Trauer_tempo']
tempo_label_dict = dict(zip(tempo_label_keys, tempo_label_values))


def create_countplots(path):
    annotaion_files = dtp.get_filelist_in_folder(path)

    for f in annotaion_files:
        f_path = path + f
        annotation_data = pd.read_csv(f_path)
        experiment_name = 'am' if annotation_data['experiment'].str.contains('(AM)').any() else 'de'
        # json_question = annotation_data['options'].apply(json.loads)

        original_label_keys = ['_A.wav', '_E.wav', '_F.wav', '_s.wav', '_W.wav', '_T.wav']
        original_label_values = ['Angst', 'Ekel', 'Freude', 'neutral', 'Wut', 'Trauer']
        original_label_dict = dict(zip(original_label_keys, original_label_values))
        pitch_label_keys  = ['A_pitch', 'E_pitch', 'F_pitch', 's_pitch', 'W_pitch', 'T_pitch']
        pitch_label_values = ['Angst_F0', 'Ekel_F0', 'Freude_F0', 'neutral_F0', 'Wut_F0', 'Trauer_F0']
        pitch_label_dict = dict(zip(pitch_label_keys, pitch_label_values))
        tempo_label_keys  = ['A_tempo', 'E_tempo', 'F_tempo', 's_tempo', 'W_tempo', 'T_tempo']
        tempo_label_values = ['Angst_tempo', 'Ekel_tempo', 'Freude_tempo', 'neutral_tempo', 'Wut_tempo', 'Trauer_tempo']
        tempo_label_dict = dict(zip(tempo_label_keys, tempo_label_values))

        experiment_plots = 'plots/' + experiment_name
        if not os.path.exists(experiment_plots):
            try:
                os.makedirs(experiment_plots)
            except OSError:
                print("Creation of the directory %s failed" % path)

        for original_label in original_label_keys:
            label_data = dtp.get_data_per_emotion(annotation_data, original_label)
            # ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            # ax = sns.barplot(x="inputvalue", y="x", data=label_data, estimator=lambda x: len(x) / label_data.shape[0] * 100)
            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            t = experiment_name + '_' + original_label_dict[original_label]
            plt.title(t)
            plt.savefig(f'{experiment_plots}/{t}.png')
            #plt.show()

        for pitch_label in pitch_label_keys:
            label_data = dtp.get_data_per_emotion(annotation_data, pitch_label)
            # ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            # ax = sns.barplot(x="inputvalue", y="inputvalue", data=label_data,estimator=lambda x: len(x) / label_data.shape[0] * 100)
            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            t = experiment_name + '_' + pitch_label_dict[pitch_label]
            plt.title(t)
            plt.savefig(f'{experiment_plots}/{t}.png')
            #plt.show()

        for tempo_label in tempo_label_keys:
            label_data = dtp.get_data_per_emotion(annotation_data, tempo_label)
            # ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            # ax = sns.barplot(x="inputvalue", y="inputvalue", data=label_data,estimator=lambda x: len(x) / label_data.shape[0] * 100)
            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])
            ax.set(ylabel="Annotation")
            ax.set(xlabel="Label")
            t = experiment_name + '_' + tempo_label_dict[tempo_label]
            plt.title(t)
            plt.savefig(f'{experiment_plots}/{t}.png')
            #plt.show()



def create_heatmap(path):

    original_label_keys = ['_A.wav', '_E.wav', '_F.wav', '_s.wav', '_W.wav', '_T.wav']
    original_label_values = ['Angst', 'Ekel', 'Freude', 'neutral', 'Wut', 'Trauer']
    original_label_dict = dict(zip(original_label_keys, original_label_values))
    pitch_label_keys = ['A_pitch', 'E_pitch', 'F_pitch', 's_pitch', 'W_pitch', 'T_pitch']
    pitch_label_values = ['Angst', 'Ekel', 'Freude', 'neutral', 'Wut', 'Trauer']
    pitch_label_dict = dict(zip(pitch_label_keys, pitch_label_values))
    tempo_label_keys = ['A_tempo', 'E_tempo', 'F_tempo', 's_tempo', 'W_tempo', 'T_tempo']
    tempo_label_values = ['Angst', 'Ekel', 'Freude', 'neutral', 'Wut', 'Trauer']
    tempo_label_dict = dict(zip(tempo_label_keys, tempo_label_values))
    all_labels_dict = {**original_label_dict, **pitch_label_dict, **tempo_label_dict}



    annotaion_files = dtp.get_filelist_in_folder(path)

    for f in annotaion_files:
        f_path = path + f
        df = pd.read_csv(f_path)
        experiment_name = 'am' if df['experiment'].str.contains('(AM)').any() else 'de'



        pat = r'({})'.format('|'.join(all_labels_dict.keys()))
        extracted = df['url'].str.extract(pat, expand=False).dropna()

        df['expected'] = extracted.apply(lambda x: all_labels_dict[x]).reindex(df.index).fillna(0)
        df.to_csv(f'normalized_labels_{experiment_name}.csv')

        df = df[df['url'].notna()]

        stimuli_groups = ['original', 'pitch', 'tempo']
        for g in stimuli_groups:
            label_keys = f'{g}_label_keys'
            df_group= df.loc[df['url'].str.contains('|'.join(eval(label_keys)))]


            confusion_matrix = pd.crosstab(df_group['expected'], df_group['inputvalue'],
                                       rownames=['Prosodischer Ausdruck'], colnames=['Annotation'],)
            sns.heatmap(confusion_matrix, annot=True, fmt='d')
            plt.title(f'{g}_{experiment_name}')
            plt.savefig(f'plots/conf_{g}_{experiment_name}.png')
            plt.show()




base_folder = 'normalized_data/'


#create_countplots(base_folder)

create_heatmap(base_folder)
