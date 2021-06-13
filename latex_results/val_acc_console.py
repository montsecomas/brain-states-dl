from utils.file_utils import load_model
import numpy as np
from nn_classification.data_loaders import SingleSubjectNNData
from nn_classification.pl_module import LitMlpClassifier, LitConvClassifier
import os.path as osp
from utils.file_utils import load_cfg
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

RAND_INIT = True
MLP = False

training_aucs = []
validation_aucs = []
training_accs = []
validation_accs = []

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
    # subject = 25
    if cfg['run_pd']:
        subject_path = f'{ckpt_paths["mlp-single-pd-subject-logs"]}/subject-{subject}/'
    else:
        subject_path = f'{ckpt_paths["mlp-single-healthy-subject-logs"]}/subject-{subject}/'

    # ---- LOAD TRAIN VAL DATASET ----
    subject_data = SingleSubjectNNData(subject=subject, classifier='mlp', cfg=cfg,
                                       read_silent_channels=True, force_read_split=True,
                                       is_cnn=(not MLP))

    for FREQ in freq_ids.keys():
        # FREQ = 'gamma'
        model_ckpt = f'ss-{FREQ}-mlp-pow{session}' if MLP else f'ss-{FREQ}-cnn-pow'
        ckpt_path = ckpt_paths[subject][model_ckpt]
        if RAND_INIT:
            if MLP:
                idx_hparams = {'n_features': subject_data.input_dataset.shape[2],
                               'n_states': 3,
                               'n_hidden_nodes': cfg['n_hidden_nodes'],
                               'n_hidden_layers': cfg['n_hidden_layers'],
                               'lr': cfg['lr'],
                               'epochs': cfg['epochs'],
                               'freq_name': FREQ,
                               'pred_feature': cfg['pred_feature'],
                               'input_dropout': cfg['input_dropout'],
                               'mlp_dropout': cfg['mlp_dropout'],
                               'weight_decay': cfg['weight_decay'],
                               'num_classes': 3}
                model = LitMlpClassifier(hparams=idx_hparams)
            else:
                idx_hparams = {'input_channels': 60,
                               'kernel_size': 3,
                               'n_states': 3,
                               'lr': cfg['lr'],
                               'epochs': cfg['epochs'],
                               'input_dropout': cfg['input_dropout'],
                               'freq_name': FREQ,
                               'num_classes': 3}

            model = LitConvClassifier(hparams=idx_hparams)
        else:
            model = load_model(osp.join(subject_path, ckpt_path), model='mlp') if MLP else \
                load_model(osp.join(subject_path, ckpt_path), model='cnn')

        freq_id = freq_ids[FREQ]
        train_loader, val_loader = subject_data.mlp_ds_loaders(freq=freq_id)  # 0 alpha, 1 beta, 2 gamma

        silent_chan = np.load(osp.join(cfg['data_path'], inputs_path, f'res_subject_{subject}',
                                       f'silent-channels-{subject}.npy'))

        dataset = ['Training', 'Validation']
        for i, loader in enumerate([train_loader, val_loader]):
            batch = next(iter(loader))
            inputs, targets = batch

            # ---- MAKE PREDICTIONS ----
            test_input_tensor = inputs.float().clone()
            out_probs = model(test_input_tensor).detach()
            out_classes = np.argmax(out_probs.numpy(), axis=1)

            acc = sum(out_classes == targets.numpy()) / len(targets)
            auc = torch.as_tensor(roc_auc_score(targets, F.softmax(out_probs, dim=-1).numpy(),
                                                average='macro', multi_class='ovo', labels=[0, 1, 2]))

            if i == 0:
                training_accs.append(round(acc.item(), 3))
                training_aucs.append(round(auc.item(), 3))
            else:
                validation_accs.append(round(acc.item(), 3))
                validation_aucs.append(round(auc.item(), 3))

PRINT_AUCS = False
PRINT_ACCS = False
PRINT_TRAIN = False
if PRINT_AUCS:
    arr = training_aucs if PRINT_TRAIN else validation_aucs
    print(f'Alpha freq: {arr[0::3]}')
    print(f'Beta freq: {arr[1::3]}')
    print(f'Gamma freq: {arr[2::3]}')
if PRINT_ACCS:
    arr = training_accs if PRINT_TRAIN else validation_accs
    print(f'Alpha freq: {arr[0::3]}')
    print(f'Beta freq: {arr[1::3]}')
    print(f'Gamma freq: {arr[2::3]}')

# ------------- SILENT CHANNELS --------
PRINT_SILENT_CHAN = False
if PRINT_SILENT_CHAN:
    for subject in subjects_list:
        silent_chan = np.load(osp.join(cfg['data_path'], inputs_path, f'res_subject_{subject}',
                                               f'silent-channels-{subject}.npy'))
        print(f'Subject {subject}, active={len(silent_chan)-np.sum(silent_chan)}, silent={np.sum(silent_chan)}')