from utils.file_utils import load_model
import numpy as np
from nn_classification.data_loaders import SingleSubjectNNData
import os.path as osp
from utils.file_utils import load_cfg
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

MLP = True
cfg = load_cfg()
ckpt_paths = load_cfg('nn_interpretability/ckpt_paths.yml')

freq_ids = dict({'alpha': 0, 'beta': 1, 'gamma': 2})
key_sufix = '_pd' if cfg['run_pd'] else ''
heatmaps_path = osp.join(cfg['outputs_path'], cfg[f'interpret_figures{key_sufix}'], cfg['interpret_heatmaps'])
if cfg['run_pd']:
    session = '-on' if cfg['model_on_med'] else '-off'
    subjects_list = cfg['pd_subjects']
    inputs_path = cfg['pd_dir']
else:
    session = ''
    subjects_list = cfg['healthy_subjects']
    inputs_path = cfg['healthy_dir']

for subject in subjects_list:
    subject = 25
    if cfg['run_pd']:
        subject_path = f'{ckpt_paths["mlp-single-pd-subject-logs"]}/subject-{subject}/'
    else:
        subject_path = f'{ckpt_paths["mlp-single-healthy-subject-logs"]}/subject-{subject}/'

    # ---- LOAD TRAIN VAL DATASET ----
    subject_data = SingleSubjectNNData(subject=subject, classifier='mlp', cfg=cfg,
                                       read_silent_channels=True, force_read_split=True)

    for FREQ in freq_ids.keys():
        # FREQ = 'gamma'
        model_ckpt = f'ss-{FREQ}-mlp-pow{session}' if MLP else f'ss-{FREQ}-cnn-pow'
        ckpt_path = ckpt_paths[subject][model_ckpt]
        model = load_model(osp.join(subject_path, ckpt_path), model='mlp')

        freq_id = freq_ids[FREQ]
        train_loader, val_loader = subject_data.mlp_ds_loaders(freq=freq_id)  # 0 alpha, 1 beta, 2 gamma

        batch = next(iter(val_loader))
        inputs, targets = batch
        silent_chan = np.load(osp.join(cfg['data_path'], inputs_path, f'res_subject_{subject}',
                                       f'silent-channels-{subject}.npy'))

        # ---- MAKE PREDICTIONS ----
        test_input_tensor = inputs.float().clone()
        out_probs = model(test_input_tensor).detach()
        out_classes = np.argmax(out_probs.numpy(), axis=1)

        acc = sum(out_classes == targets.numpy()) / len(targets)
        auc = torch.as_tensor(roc_auc_score(targets, F.softmax(out_probs, dim=-1).numpy(),
                                            average='macro', multi_class='ovo', labels=[0, 1, 2]))

        print(f'Validation AUC: {round(auc.item(), 3)}, Accuracy: {round(acc, 3)}')

