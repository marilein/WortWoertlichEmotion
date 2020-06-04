import os
from datetime import datetime

import numpy as np
import pandas as pd

path_de = 'raw_data/2278116_2020-05-29_11-24-14.txt'
path_am = 'raw_data/2287306_2020-05-29_11-24-48.txt'

data_de = pd.read_csv(path_de, sep='\t')
data_de.to_csv('wowoemotion_de.csv', sep=';')

data_am = pd.read_csv(path_am, sep='\t')
data_am.to_csv('wowoemotion_am.csv', sep=';')

def create_experiment_overview(df_experiment, language):
    base_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    overview = pd.DataFrame(columns=['session', 'sex', 'age'])

    sessions = df_experiment['sessionid'].unique()
    print(sessions, len(sessions))

    for session in sessions:
        session_data = df_experiment.loc[df_experiment['sessionid']==session]
        session_length = session_data.shape[0]
        if session_length == 151:
            session_overview = pd.DataFrame(columns=['session', 'sex', 'age'])
            session_overview['session'] = session_data['sessionid'].unique()
            session_overview['sex'] = session_data['sex'].unique()
            session_overview['age'] = session_data['participantage'].unique()

            overview = pd.concat([overview, session_overview])

    overview.to_csv('overview/'+base_name + 'overview_' + language + '.csv', sep=';')







create_experiment_overview(data_de, 'de')
create_experiment_overview(data_am, 'am')