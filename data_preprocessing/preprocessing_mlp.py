import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import numpy as np
from utils.file_utils import load_cfg, is_pd_patient, processed_data_path, processed_labels_path
from data_preprocessing.preprocess_module import BrainStatesSubject, BrainStatesFeaturing


if __name__ == '__main__':

    cfg = load_cfg()
    run_pd = cfg['run_pd']
    subjects_list = cfg['pd_subjects'] if run_pd else cfg['healthy_subjects']

    for subject in subjects_list:
        # subject = 25, healthy
        # subject = 55, PD
        is_pd = is_pd_patient(subject, healthy_subjects=cfg['healthy_subjects'], pd_subjects=cfg['pd_subjects'])

        sample = BrainStatesSubject(i_sub=subject, PD=is_pd, cfg=cfg)
        flat_data, flat_pks, is_pd, invalid_ch = sample.run_pipeline()
        np.save(processed_data_path(subject_id=subject, is_pd=is_pd, use_silent_channels=False,
                                    feature_name='silent_channels', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                    healthy_dir=cfg['healthy_dir']),
                invalid_ch)

        sample_featuring = BrainStatesFeaturing(input_ts=flat_data, input_labels=flat_pks, pd_sub=is_pd,
                                                use_silent_channels=cfg['use_silent_channels'])

        if not cfg['use_silent_channels']:
            signal_ds = sample_featuring.build_signal_dataset()
            # cov_ds = sample_featuring.build_cov_dataset()
            # cor_ds = sample_featuring.build_cor_dataset()

            prefix = 'pow_' if (cfg['mat_dict'] == 'dataSorted') else 'ic_'

            print('Saving output datasets')
            np.save(processed_data_path(subject_id=subject, is_pd=is_pd, use_silent_channels=cfg['use_silent_channels'],
                                        feature_name=f'{prefix}mean', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                        healthy_dir=cfg['healthy_dir']),
                    signal_ds)
            # np.save(processed_data_path(subject_id=subject, is_pd=is_pd, use_silent_channels=cfg['use_silent_channels'],
            #                             feature_name=f'{prefix}cov', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
            #                             healthy_dir=cfg['healthy_dir']),
            #         cov_ds)
            # np.save(processed_data_path(subject_id=subject, is_pd=is_pd, use_silent_channels=cfg['use_silent_channels'],
            #                             feature_name=f'{prefix}cor', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
            #                             healthy_dir=cfg['healthy_dir']),
            #         cor_ds)
            np.save(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=cfg['data_path'],
                                          pd_dir=cfg['pd_dir'], healthy_dir=cfg['healthy_dir']),
                    sample_featuring.ts_labels)
        else:
            signal_ds = sample_featuring.build_signal_dataset()
            print('Saving output datasets')
            np.save(processed_data_path(subject_id=subject, is_pd=is_pd, use_silent_channels=cfg['use_silent_channels'],
                                        feature_name='pow_mean', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                        healthy_dir=cfg['healthy_dir']),
                    signal_ds)
            np.save(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=cfg['data_path'],
                                          pd_dir=cfg['pd_dir'], healthy_dir=cfg['healthy_dir']),
                    sample_featuring.ts_labels)
