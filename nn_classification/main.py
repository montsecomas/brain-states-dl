'''
Training options:
- plit_subj_mlp_train
    for each subject in healthy subjects, train a model (90%-10% split)
    idx_hparams = {'n_features': input_data.shape[2],
                           'n_states': len(np.unique(targets)),
                           'n_hidden_nodes': cfg['n_hidden_nodes'],
                           'n_hidden_layers': cfg['n_hidden_layers'],
                           'lr': cfg['lr'],
                           'epochs': cfg['epochs'],
                           'freq_name': freqs[freq],
                           'pred_feature': cfg['pred_feature'],
                           'input_dropout': None,
                           'mlp_dropout': None,
                           'weight_decay': cfg['weight_decay'],
                           'num_classes': 3}
    model = LitMlpClassifier(hparams=idx_hparams)
    batch_size: 64
    (works for ICA and power-values)
    (prints confusion matrices)

- plit_crossval_mlp_train
    uses all subjects in config file (healthy) but one for training, and that left subject for validation
    idx_hparams = {'n_features': input_data.shape[2],
                           'n_states': len(np.unique(targets)),
                           'n_hidden_nodes': cfg['n_hidden_nodes'],
                           'n_hidden_layers': cfg['n_hidden_layers'],
                           'lr': cfg['lr'],
                           'epochs': cfg['epochs'],
                           'freq_name': freqs[freq],
                           'pred_feature': cfg['pred_feature'],
                           'input_dropout': cfg['input_dropout'],
                           'mlp_dropout': cfg['mlp_dropout'],
                           'weight_decay': cfg['weight_decay'],
                           'num_classes': 3}

            model = LitMlpClassifier(hparams=idx_hparams)
    (works for power-values and prints confusion matrices)
    batch_size: 128
    n_hidden_nodes : 128
    n_hidden_layers: 2
    lr: 0.001
    epochs: 80
    input_dropout: 0.2 # >0 or None

- plit_subj_conv_train
- plit_crossval_conv_train
'''