import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines


import numpy as np
from utils.utils import load_cfg, is_pd_patient, processed_data_path, processed_labels_path
from data_preprocessing.preprocess_module import BrainStatesSubject, BrainStatesFeaturing


if __name__ == '__main__':

    cfg = load_cfg()

    for subject in cfg['healthy_subjects']:
        # subject = 25
        is_pd = is_pd_patient(subject, healthy_subjects=cfg['healthy_subjects'], pd_subjects=cfg['pd_subjects'])

        sample = BrainStatesSubject(i_sub=subject, PD=is_pd, subset=cfg['mat_dict'],
                                    data_path=cfg['data_path'], pd_dir=cfg['pd_dir'], healthy_dir=cfg['healthy_dir'],
                                    clean_channels=cfg['classic_prepro'])
        flat_data, flat_pks, is_pd = sample.run_pipeline()

        sample_featuring = BrainStatesFeaturing(input_ts=flat_data, input_labels=flat_pks, pd_sub=is_pd,
                                                clean_channels=cfg['classic_prepro'])

        if cfg['classic_prepro']:
            signal_ds = sample_featuring.build_signal_dataset()
            cov_ds = sample_featuring.build_cov_dataset()
            cor_ds = sample_featuring.build_cor_dataset()
            # 25: signal_ds.shape, cov_ds.shape, cov_ds.shape = ((3, 1296, 42), (3, 1296, 861), (3, 1296, 861))
            # 26: signal_ds.shape, cov_ds.shape, cov_ds.shape = ((3, 1017, 49), (3, 1017, 1176), (3, 1017, 1176))

            print('Saving output datasets')
            np.save(processed_data_path(subject_id=subject, is_pd=is_pd, classic_prepro=cfg['classic_prepro'],
                                        feature_name='pow_mean', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                        healthy_dir=cfg['healthy_dir']),
                    signal_ds)
            np.save(processed_data_path(subject_id=subject, is_pd=is_pd, classic_prepro=cfg['classic_prepro'],
                                        feature_name='pow_cov', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                        healthy_dir=cfg['healthy_dir']),
                    cov_ds)
            np.save(processed_data_path(subject_id=subject, is_pd=is_pd, classic_prepro=cfg['classic_prepro'],
                                        feature_name='pow_cor', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                        healthy_dir=cfg['healthy_dir']),
                    cor_ds)
            np.save(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=cfg['data_path'],
                                          pd_dir=cfg['pd_dir'], healthy_dir=cfg['healthy_dir']),
                    sample_featuring.ts_labels)
        else:
            signal_ds = sample_featuring.build_signal_dataset()
            print('Saving output datasets')
            np.save(processed_data_path(subject_id=subject, is_pd=is_pd, classic_prepro=cfg['classic_prepro'],
                                        feature_name='pow_mean', data_path=cfg['data_path'], pd_dir=cfg['pd_dir'],
                                        healthy_dir=cfg['healthy_dir']),
                    signal_ds)
            np.save(processed_labels_path(subject_id=subject, is_pd=is_pd, data_path=cfg['data_path'],
                                          pd_dir=cfg['pd_dir'], healthy_dir=cfg['healthy_dir']),
                    sample_featuring.ts_labels)
