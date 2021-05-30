import sys
sys.path.append('/Users/mcomastu/TFM/brain-states-dl') # TODO: remove this first two lines

from utils.file_utils import load_cfg
from nn_interpretability.subj_apply_interpret import run_attributions_bloc


if __name__ == '__main__':
    cfg = load_cfg()
    ckpt_paths = load_cfg('nn_interpretability/ckpt_paths.yml')
    DO_HISTS = False
    DO_TABLES = True
    DO_PLOT_MAP = True
    DO_SURF_MAP = True
    USE_SILENT_CHANNELS = True
    # 'IntegratedGradients', 'ShapleyValueSampling', 'KernelShap', 'Lime'
    METHOD = 'ShapleyValueSampling'

    run_attributions_bloc(cfg, ckpt_paths, draw_histograms=DO_HISTS, save_tables=DO_TABLES,
                          draw_dots_map=DO_PLOT_MAP, draw_surf_map=DO_SURF_MAP, method=METHOD,
                          nn_use_silent_channels=USE_SILENT_CHANNELS)
