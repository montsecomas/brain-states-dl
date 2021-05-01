import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import numpy as np
from utils.utils import load_cfg, is_pd_patient, processed_data_path, processed_labels_path
from data_preprocessing.preprocess_module import BrainStatesSubject, BrainStatesFeaturing


if __name__ == '__main__':

    cfg = load_cfg()

    for subject in cfg['healthy_subjects']:
        # subject = 26
        is_pd = is_pd_patient(subject, healthy_subjects=cfg['healthy_subjects'], pd_subjects=cfg['pd_subjects'])

        sample = BrainStatesSubject(i_sub=subject, PD=is_pd, subset=cfg['mat_dict'],
                                    data_path=cfg['data_path'], pd_dir=cfg['pd_dir'], healthy_dir=cfg['healthy_dir'],
                                    use_silent_channels=cfg['use_silent_channels'])
        flat_data, flat_pks, is_pd = sample.run_pipeline()

        sample_featuring = BrainStatesFeaturing(input_ts=flat_data, input_labels=flat_pks, pd_sub=is_pd,
                                                use_silent_channels=cfg['use_silent_channels'])

        rs_freqs = sample_featuring.bandpassed.transpose((1, 2, 0, 3))
        flat_freqs = rs_freqs.reshape(*rs_freqs.shape[:2], -1)
        fin_freqs = flat_freqs.transpose((0, 2, 1))

        print('Saving output datasets')
        np.save(processed_data_path(subject_id=subject, is_pd=is_pd, data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                    healthy_dir=cfg['healthy_dir'], conv=True),
                fin_freqs)
        np.save(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=cfg['data_path'],
                                      pd_dir=cfg['pd_dir'], healthy_dir=cfg['healthy_dir']),
                sample_featuring.ts_labels)
