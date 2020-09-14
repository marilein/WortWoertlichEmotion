import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_processor as dtp
import matplotlib.ticker as ticker



original_label_keys = ['_A.wav', '_E.wav', '_F.wav', '_s.wav', '_W.wav', '_T.wav']
original_label_values = ['Angst', 'Ekel', 'Freude', 'neutral', 'Wut', 'Trauer']
original_label_dict = dict(zip(original_label_keys, original_label_values))
pitch_label_keys  = ['A_pitch', 'E_pitch', 'F_pitch', 's_pitch', 'W_pitch', 'T_pitch']
pitch_label_values = ['Angst_F0', 'Ekel_F0', 'Freude_F0', 'neutral_F0', 'Wut_F0', 'Trauer_F0']
pitch_label_dict = dict(zip(pitch_label_keys, pitch_label_values))
tempo_label_keys  = ['A_tempo', 'E_tempo', 'F_tempo', 's_tempo', 'W_tempo', 'T_tempo']
tempo_label_values = ['Angst_tempo', 'Ekel_tempo', 'Freude_tempo', 'neutral_tempo', 'Wut_tempo', 'Trauer_tempo']
tempo_label_dict = dict(zip(tempo_label_keys, tempo_label_values))


def split_speaker_data(label_data):
    # returns a list of dataframes, each of those contains annotations of one speaker recordings
    label_data = label_data[label_data['url'].notna()]
    m_data = label_data.loc[label_data["url"].str.contains("M_")]
    f_data = label_data.loc[~label_data.index.isin(m_data.index)]

    return [m_data, f_data]

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
            speaker_data_list = split_speaker_data(label_data)
            for speaker_data in speaker_data_list:
                annot_count = speaker_data.shape[0]
                # ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
                # ax = sns.barplot(x="inputvalue", data=label_data, estimator=lambda x: len(x) / label_data.shape[0] * 100)

                ax = sns.countplot(x="inputvalue", data=speaker_data,
                                   order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])

                ax.set(ylabel="Anzahl Annotationen")
                ax.set(xlabel="Label")

                # Make twin axis
                ax2 = ax.twinx()

                # Switch so count axis is on right, frequency on left
                ax2.yaxis.tick_left()
                ax.yaxis.tick_right()

                # Also switch the labels over
                ax.yaxis.set_label_position('right')
                ax2.yaxis.set_label_position('left')

                ax2.set_ylabel('Annotationen [%]')

                for p in ax.patches:
                    x = p.get_bbox().get_points()[:, 0]
                    y = p.get_bbox().get_points()[1, 1]
                    ax.annotate('{:.1f}%'.format(100. * y / annot_count), (x.mean(), y),
                                ha='center', va='bottom')  # set the alignment of the text

                # Use a LinearLocator to ensure the correct number of ticks
                ax.yaxis.set_major_locator(ticker.LinearLocator(11))

                # Fix the frequency range to 0-100
                ax2.set_ylim(0, 100)
                ax.set_ylim(0, annot_count)

                # And use a MultipleLocator to ensure a tick spacing of 10
                ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

                # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
                ax2.grid(None)

                speaker = "F_"

                if speaker_data["url"].str.contains("M_").any():
                    speaker = 'M_'


                t = experiment_name + '_' + speaker + original_label_dict[original_label]
                plt.title(t)
                plt.savefig(f'{experiment_plots}/{t}.png')
                plt.autoscale()
                plt.show()
                plt.clf()



        for pitch_label in pitch_label_keys:
            label_data = dtp.get_data_per_emotion(annotation_data, pitch_label)
            # ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            # ax = sns.barplot(x="inputvalue", y="inputvalue", data=label_data,estimator=lambda x: len(x) / label_data.shape[0] * 100)
            annot_count = label_data.shape[0]
            # ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            # ax = sns.barplot(x="inputvalue", data=label_data, estimator=lambda x: len(x) / label_data.shape[0] * 100)

            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])

            ax.set(ylabel="Anzahl Annotationen")
            ax.set(xlabel="Label")

            # Make twin axis
            ax2 = ax.twinx()

            # Switch so count axis is on right, frequency on left
            ax2.yaxis.tick_left()
            ax.yaxis.tick_right()

            # Also switch the labels over
            ax.yaxis.set_label_position('right')
            ax2.yaxis.set_label_position('left')

            ax2.set_ylabel('Annotationen [%]')

            for p in ax.patches:
                x = p.get_bbox().get_points()[:, 0]
                y = p.get_bbox().get_points()[1, 1]
                ax.annotate('{:.1f}%'.format(100. * y / annot_count), (x.mean(), y),
                            ha='center', va='bottom')  # set the alignment of the text

            # Use a LinearLocator to ensure the correct number of ticks
            ax.yaxis.set_major_locator(ticker.LinearLocator(11))

            # Fix the frequency range to 0-100
            ax2.set_ylim(0, 100)
            ax.set_ylim(0, annot_count)

            # And use a MultipleLocator to ensure a tick spacing of 10
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

            # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
            ax2.grid(None)
            t = experiment_name + '_' + pitch_label_dict[pitch_label]
            plt.title(t)
            plt.savefig(f'{experiment_plots}/{t}.png')
            plt.clf()
            #plt.show()

        for tempo_label in tempo_label_keys:
            label_data = dtp.get_data_per_emotion(annotation_data, tempo_label)

            # ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            # ax = sns.barplot(x="inputvalue", y="inputvalue", data=label_data,estimator=lambda x: len(x) / label_data.shape[0] * 100)
            annot_count = label_data.shape[0]
            # ax = sns.catplot(x="inputvalue", kind="count", palette="ch:.25", data=label_data);
            # ax = sns.barplot(x="inputvalue", data=label_data, estimator=lambda x: len(x) / label_data.shape[0] * 100)

            ax = sns.countplot(x="inputvalue", data=label_data,
                               order=['Freude', 'Trauer', 'Wut', 'Angst', 'Ekel', 'neutral'])

            ax.set(ylabel="Anzahl Annotationen")
            ax.set(xlabel="Label")

            # Make twin axis
            ax2 = ax.twinx()

            # Switch so count axis is on right, frequency on left
            ax2.yaxis.tick_left()
            ax.yaxis.tick_right()

            # Also switch the labels over
            ax.yaxis.set_label_position('right')
            ax2.yaxis.set_label_position('left')

            ax2.set_ylabel('Annotationen [%]')

            for p in ax.patches:
                x = p.get_bbox().get_points()[:, 0]
                y = p.get_bbox().get_points()[1, 1]
                ax.annotate('{:.1f}%'.format(100. * y / annot_count), (x.mean(), y),
                            ha='center', va='bottom')  # set the alignment of the text

            # Use a LinearLocator to ensure the correct number of ticks
            ax.yaxis.set_major_locator(ticker.LinearLocator(11))

            # Fix the frequency range to 0-100
            ax2.set_ylim(0, 100)
            ax.set_ylim(0, annot_count)

            # And use a MultipleLocator to ensure a tick spacing of 10
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

            # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
            ax2.grid(None)
            t = experiment_name + '_' + tempo_label_dict[tempo_label]
            plt.title(t)
            plt.savefig(f'{experiment_plots}/{t}.png')
            #plt.show()#
            plt.clf()



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

            ft = lambda x, pos: '{:.0%}'.format(x)

            confusion_matrix = pd.crosstab(df_group['expected'], df_group['inputvalue'],
                                       rownames=['Prosodischer Ausdruck'], colnames=['Annotation'],)
            sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot=True,
                        fmt='0.1%', cmap = sns.light_palette("navy"), cbar_kws={'format': ticker.FuncFormatter(ft)})
            #cbar_kws={'format': '%.0f%%', 'ticks': [0,100]}
            plt.title(f'{g}_{experiment_name}')
            plt.savefig(f'plots/conf_{g}_{experiment_name}.png')
            plt.show()



def analyze_pitch():
    f0_aggregation = pd.read_csv('f0_aggregation.csv')
    f0_m = f0_aggregation.loc[f0_aggregation['Geschlecht']=='männlich']
    f0_f = f0_aggregation.loc[f0_aggregation['Geschlecht'] == 'weiblich']

    f0_aggregation = f0_aggregation.sort_values(by=['Emotion', 'Geschlecht'])
    '''
    sns.relplot(x="sound", y="f0_mean", hue="emotion", size="sex",
                sizes=(40, 400), alpha=.5, palette="muted",
                height=6, data=f0_aggregation)
    plt.savefig('plots/f0/f0_mean.png')
    '''
    g = sns.relplot(x="sound", y="f0_mean", hue="Emotion",col="Emotion",data=f0_aggregation, style='Geschlecht',
                     size='Geschlecht', sizes=(100, 50),row_order=['Emotion'], col_wrap=3)
    g.set_ylabels('F0 (Hz)')
    g.set_xlabels('Stimulus')
    #g.axes.set_xticks(range(1, len(f0_aggregation['Emotion'].unique()) + 1))
    #g.set_xticklabels(f0_aggregation['Emotion'].unique(), step=6, rotation=30)
    plt.xticks('')

    g.fig.suptitle('F0-Durchschnittswerte der Originalsprachaufnahmen ')
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    g.fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    #plt.title('F0-Durchschnittswerte der Originalsprachaufnahmen')
    plt.savefig('plots/f0/f0_mean_per_emotion_sorted_sex.png')

    plt.clf()

    g = sns.relplot(x="sound", y="f0_median", hue="Emotion", col="Geschlecht", data=f0_aggregation)
    g.set_ylabels('F0 (Hz)')
    g.set_xlabels('Stimulus')
    plt.xticks('')

    g.fig.suptitle('F0-Medianwerte der Originalsprachaufnahmen ')
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    g.fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    # plt.title('F0-Durchschnittswerte der Originalsprachaufnahmen')
    plt.savefig('plots/f0/f0_median_two_in_one.png')

    plt.clf()

    sns.relplot(x="sound", y="f0_mean", hue="Emotion", palette="muted",
                height=6, data=f0_m)
    plt.ylabel('F0 (Hz)')
    plt.xlabel ('Stimulus')
    plt.xticks('')
    plt.savefig('plots/f0/f0_mean_m.png')

    plt.clf()

    sns.relplot(x="sound", y="f0_mean", hue="Emotion", palette="muted",
                height=6, data=f0_f)
    plt.ylabel('F0 (Hz)')
    plt.xlabel('Stimulus')
    plt.xticks('')
    plt.savefig('plots/f0/f0_mean_f.png')
    plt.clf()




base_folder = 'normalized_data/'


#create_countplots(base_folder)

#create_heatmap(base_folder)

analyze_pitch()



