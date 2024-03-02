import torch
from util import set_seed
from util import compute_age_mae
from main_dyn_loc_rep import load_data
from models.resnet3d import SupConResNet


class opts:
    trial = 0
    fold = 0
    modality = 'stiffness'

    save_path = '/rds/user/jnt27/hpc-work/contrastive-brain-age-MREoutput/brain-age-mri/models/'

    # stiffness baseline
    file = ('pretrained_expw_stiffness_threshold_E50_trial' + str(trial) + '_fold' + str(fold) +
            '_NN_nb_selection_similarity_end_NN_nb_10_NN_nb_step_size_0/ckpt_epoch_50.pth')

    save_file = save_path + file

    batch_size = 32
    stratify_bins = 0
    device = 'cuda'
    location = 'cluster'
    n_views = 2
    tf = 'none'
    model = 'resnet18'


if __name__ == '__main__':

    opts = opts()

    for trial in range(5):
        opts.trial = trial
        opts.file = ('NEW_pretrained_expw_stiffness_expw_E50_trial' + str(trial) +
                     '_fold0_NN_nb_selection_similarity_end_NN_nb_14_NN_nb_step_size_1')
        opts.save_file = opts.save_path + opts.file + '/ckpt_epoch_50.pth'

        set_seed(opts.trial)

        train_loader, train_loader_score, test_loader = load_data(opts)

        model = SupConResNet(opts.model, feat_dim=128)

        print(f"Using {opts.save_file}")
        state = torch.load(opts.save_file)

        model.load_state_dict(state['model'])
        del state

        if opts.device == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
            model = torch.nn.DataParallel(model)

        model = model.to(opts.device)

        mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        print(f"MAE on train: {mae_train:.2f}, MAE on test: {mae_test:.2f}")
        print(f"Trial {opts.trial} done")
