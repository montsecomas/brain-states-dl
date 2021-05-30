import yaml
import os.path as osp
from nn_classification.pl_module import LitMlpClassifier, LitConvClassifier


def load_cfg(path="config.yaml"):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def is_pd_patient(i_sub, healthy_subjects, pd_subjects):
    if i_sub in healthy_subjects:
        return False
    elif i_sub in pd_subjects:
        return True
    else:
        raise ValueError(('No data for subject ' + str(i_sub)))


def raw_input_path(subject_id, is_pd, data_path, pd_dir, healthy_dir):
    if is_pd:
        return osp.join(data_path, pd_dir, f"dataClean-ICA-{subject_id}-T1.mat")
    else:
        return osp.join(data_path, healthy_dir, f"dataClean-ICA3-{subject_id}-T1.mat")


def subject_res_dir(subject_id, is_pd, data_path, pd_dir, healthy_dir):
    if is_pd:
        return osp.join(data_path, pd_dir, f"res_subject_{subject_id}")
    else:
        return osp.join(data_path, healthy_dir, f"res_subject_{subject_id}")


def processed_data_path(subject_id, is_pd, data_path, pd_dir, healthy_dir,
                        use_silent_channels=None, feature_name=None, conv=False):
    """

    :param subject_id: subject id
    :param is_pd: True or False
    :param use_silent_channels:
    :param feature_name: values: 'pow_mean, pow_cov, pow_cor, ica_mean, ica_cov, ica_cor'
    :param data_path:
    :param pd_dir:
    :param healthy_dir:
    :return: string
    """
    output_path = subject_res_dir(subject_id, is_pd, data_path, pd_dir, healthy_dir)
    if conv:
        return osp.join(output_path, f"raw-freqs-{subject_id}.npy")
    else:
        sufix = '-all-channels' if use_silent_channels else ''

        if feature_name == 'pow_mean':
            return osp.join(output_path, f"freq-pow-mean-{subject_id}{sufix}.npy")
        elif feature_name == 'pow_cov':
            return osp.join(output_path, f"freq-pow-cov-{subject_id}.npy")
        elif feature_name == 'pow_cor':
            return osp.join(output_path, f"freq-pow-cor-{subject_id}.npy")
        elif feature_name == 'ic_mean':
            return osp.join(output_path, f"freq-ica-mean-{subject_id}.npy")
        elif feature_name == 'ic_cov':
            return osp.join(output_path, f"freq-ica_cov-{subject_id}.npy")
        elif feature_name == 'ic_cor':
            return osp.join(output_path, f"freq-ica-cor-{subject_id}.npy")
        elif feature_name == 'silent_channels':
            return osp.join(output_path, f"silent-channels-{subject_id}.npy")


def processed_labels_path(subject_id, is_pd, data_path, pd_dir, healthy_dir):
    output_path = subject_res_dir(subject_id, is_pd, data_path, pd_dir, healthy_dir)
    return osp.join(output_path, f"labels-{subject_id}.npy")


def load_model(ckpt_path, model='mlp'):

    if model == 'mlp':
        LitClassifier = LitMlpClassifier
    elif model == 'cnn':
        LitClassifier = LitConvClassifier

    model = LitClassifier.load_from_checkpoint(checkpoint_path=ckpt_path)

    return model





