import numpy as np
import pandas as pd
import glob
from utils.file_utils import load_cfg


cfg = load_cfg()
healthy_subjects = [25, 26, 27, 30]
pd_patients = [55, 58, 59, 61, 64, 67]
m_data_hs = [[], [], []]
m_data_pd = [[], [], []]

for i, subjects_list in enumerate([healthy_subjects, pd_patients]):
    for subject in subjects_list:
        # subject = 25
        folder_sf = '-healthy' if subject in healthy_subjects else '-pd'
        med = '' if subject in healthy_subjects else '-on'

        imp_paths = glob.glob(f'data_outputs/mlp_importances{folder_sf}/{subject}{med}*gamma.csv')
        imp_paths.sort()

        for motiv in np.arange(3):
            # motiv = 0
            m_path = imp_paths[motiv]

            m_data = pd.read_csv(m_path).iloc[:, :-1]
            m_data.columns = ['electrode_id', 'electrode_name', 'is_silent', 'mean_contr']
            m_data['subject_id'] = subject
            filtered_data = m_data[m_data.is_silent == False].sort_values('mean_contr', ascending=False)
            filtered_data['cum_contr'] = filtered_data.groupby('subject_id')['mean_contr'].transform(pd.Series.cumsum)

            if i == 0:
                m_data_hs[motiv].append(filtered_data)
            else:
                m_data_pd[motiv].append(filtered_data)


print('\nNumber of features for 50% contribution and 90&\n\n')
for j, subjects_list in enumerate([healthy_subjects, []]):
# for j, subjects_list in enumerate([healthy_subjects, pd_patients]):
    for i, subject in enumerate(subjects_list):
        n = len(m_data_hs[0][i]) if j == 0 else len(m_data_pd[0][i])
        print(f'--------------\nSUBJECT {subject}, {n} electrodes\n-------------- ')
        for motiv in np.arange(3):
            print(f'--------------\nMOTIVATION {motiv}\n--------------')
            mk = m_data_hs[motiv][i] if j == 0 else m_data_pd[motiv][i]
            for cum_contribution in [0.1, 0.3, 0.5, 0.7, 0.9]:
                print(f'Cumulative contribution {round(cum_contribution*100, 0)}, '
                      f'{len(mk[mk.cum_contr<=cum_contribution])} electrodes')

