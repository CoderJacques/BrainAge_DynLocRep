import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import os


def read_data(path, dataset):
    print(f"Read {dataset.upper()}")
    df = pd.read_csv(os.path.join(path + '/' + dataset + '_labels/', dataset + "_participants.tsv"), sep="\t")
    df['participant_id'] = df['participant_id'].astype(str)

    if dataset == "train":
        x_arr = np.load(os.path.join(path + '/' + dataset + '_quasiraw/' + dataset + '_quasiraw/',
                                     dataset + "_quasiraw_2mm.npy"), mmap_mode="r")
        participants_id = np.load(os.path.join(path + '/' + dataset + '_quasiraw/' + dataset + '_quasiraw/',
                                               "participants_id.npy"))
    elif dataset == "val":
        x_arr = np.load(os.path.join(path + '/' + dataset + '_quasiraw/', dataset + "_quasiraw_2mm.npy"), mmap_mode="r")
        participants_id = np.load(os.path.join(path + '/' + dataset + '_quasiraw/', "participants_id.npy"))
    else:
        raise ValueError("Invalid dataset")

    matching_ages = df[df['participant_id'].isin(participants_id)][['participant_id', 'age', 'site']]
    y_arr = matching_ages[['age', 'site']].values

    print("- y size [original]:", y_arr.shape)
    print("- x size [original]:", x_arr.shape)
    assert y_arr.shape[0] == x_arr.shape[0]

    return x_arr, y_arr


class OpenBHB(torch.utils.data.Dataset):
    """from https://github.com/EIDOSLAB/contrastive-brain-age-prediction"""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train

        dataset = "train" if train else "val"

        self.X, self.y = read_data(root + '/' + dataset, dataset)
        self.T = transform

        print(f"Read {len(self.X)} records")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        if self.T is not None:
            x = self.T(x)

        # sample, age, site
        age, site = y[0], y[1]

        return x, age, site


class MREDataset(torch.utils.data.Dataset):
    def __init__(self, modality, transform=None, train=True, location='local', fold=0):

        (stiffness, dr, T1, age, sex, study,
         id, imbalance_percentages, MRE_coverage) = load_samples(study='ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA',
                                                                 location=location)

        if modality == 'stiffness':
            _, mu_stiff, sigma_stiff = normalize_mean_0_std_1(stiffness, default_value=0, mu_nonzero=None,
                                                              sigma_nonzero=None)
            [self.mu, self.sigma] = mu_stiff, sigma_stiff

        elif modality == 'dr':
            _, mu_dr, sigma_dr = normalize_mean_0_std_1(dr, default_value=0, mu_nonzero=None, sigma_nonzero=None)
            [self.mu, self.sigma] = mu_dr, sigma_dr

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        assert fold in range(5) or fold is None

        if fold in range(5):
            for fold_iter, (train_ids, test_ids) in enumerate(kfold.split(stiffness)):

                if fold_iter == fold:
                    stiffness_train, stiffness_test = stiffness[train_ids], stiffness[test_ids]
                    dr_train, dr_test = dr[train_ids], dr[test_ids]
                    T1_train, T1_test = T1[train_ids], T1[test_ids]
                    sex_train, sex_test = sex[train_ids], sex[test_ids]
                    age_train, age_test = age[train_ids], age[test_ids]
                    study_train, study_test = study[train_ids], study[test_ids]
                    imbalance_train, imbalance_test = imbalance_percentages[train_ids], imbalance_percentages[test_ids]
                    MRE_coverage_train, MRE_coverage_test = MRE_coverage[train_ids], MRE_coverage[test_ids]

                else:
                    continue
        else:
            stiffness_train, stiffness_test, \
                dr_train, dr_test, \
                T1_train, T1_test, \
                age_train, age_test, \
                sex_train, sex_test, \
                study_train, study_test, \
                imbalance_train, imbalance_test, \
                MRE_coverage_train, MRE_coverage_test = train_test_split(stiffness, dr, T1, age, sex, study,
                                                                         imbalance_percentages, MRE_coverage,
                                                                         test_size=0.2,
                                                                         random_state=42)

        if train:
            self.y = age_train
            self.sex = sex_train
            self.site = study_train
            self.imbalance = imbalance_train
            self.MRE_coverage = MRE_coverage_train

            if modality == 'stiffness':
                self.x = stiffness_train
            elif modality == 'dr':
                self.x = dr_train
            elif modality == 'T1':
                self.x = T1_train

        else:
            self.y = age_test
            self.sex = sex_test
            self.site = study_test
            self.imbalance = imbalance_test
            self.MRE_coverage = MRE_coverage_test

            if modality == 'stiffness':
                self.x = stiffness_test
            elif modality == 'dr':
                self.x = dr_test
            elif modality == 'T1':
                self.x = T1_test

        self.modality = modality
        self.T = transform

    def norm(self):

        default_value = 0

        if self.modality == 'T1':
            self.x = norm_whole_batch(self.x, 'mean_std', default_value)

        elif self.modality == 'dr' or self.modality == 'stiffness':
            self.x, _, _ = normalize_mean_0_std_1(self.x,
                                                  default_value=default_value,
                                                  mu_nonzero=self.mu,
                                                  sigma_nonzero=self.sigma)

        else:
            raise ValueError('Invalid modality')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = self.x[index]
        y = self.y[index]

        sex = self.sex[index]
        site = self.site[index]
        imbalance = self.imbalance[index]
        MRE_coverage = self.MRE_coverage[index]

        if self.T is not None:
            x = self.T(x)
        else:
            x = torch.from_numpy(x).float()

        return x, y, (sex, site, imbalance, MRE_coverage)


def norm_whole_batch(batch, norm, default_value):
    batch_normed = np.zeros_like(batch)

    for i in range(batch.shape[0]):
        if norm == 'mean_std':
            batch_normed[i], _, _ = normalize_mean_0_std_1(batch[i], default_value, None, None)

        else:
            raise ValueError('norm has to be min_max or mean_std')

    return batch_normed


def normalize_mean_0_std_1(arr, default_value, mu_nonzero, sigma_nonzero):
    arr_nonzero = arr[np.nonzero(arr)]

    if mu_nonzero is None and sigma_nonzero is None:
        mu_nonzero = np.mean(arr_nonzero)
        sigma_nonzero = np.std(arr_nonzero)

    if default_value == 0:
        arr_pp = np.zeros_like(arr)

    elif default_value == -1:
        arr_pp = np.ones_like(arr) * -1

    else:
        raise ValueError('default_value has to be 0 or -1')

    arr_pp[np.nonzero(arr)] = (arr[np.nonzero(arr)] - mu_nonzero) / sigma_nonzero

    return arr_pp, mu_nonzero, sigma_nonzero


def load_samples(study, location):
    if location == 'local':
        prefix_path = '/Users/jakobtraeuble/PycharmProjects/BrainAgeMRE/Data'

    elif location == 'cluster':
        prefix_path = '/home/jnt27/rds/hpc-work/MRE'

    else:
        raise ValueError('specific location as local or cluster')

    stiffness_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/stiffness_134.npy', allow_pickle=True)
    dr_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/dr_134.npy', allow_pickle=True)
    T1_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/T1_masked_ATLAS.npy', allow_pickle=True)  # T1_ATLAS.npy
    age_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/age_ATLAS.npy', allow_pickle=True)
    sex_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/sex_ATLAS.npy', allow_pickle=True)
    id_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/id_ATLAS.npy', allow_pickle=True)
    MRE_coverage_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/MRE_coverage_ATLAS.npy', allow_pickle=True)
    study_ATLAS = np.array(['ATLAS'] * len(age_ATLAS))

    stiffness_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/stiffness_OA.npy', allow_pickle=True)
    dr_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/dr_OA.npy', allow_pickle=True)
    T1_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/T1_masked_OA.npy', allow_pickle=True)  # T1_OA.npy
    age_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/age_OA.npy', allow_pickle=True)
    sex_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/sex_OA.npy', allow_pickle=True)
    id_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/id_OA.npy', allow_pickle=True)
    MRE_coverage_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/MRE_coverage_OA.npy', allow_pickle=True)
    study_OA = np.array(['CN'] * len(age_OA))

    stiffness_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/stiffness_BMI.npy', allow_pickle=True)
    dr_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/dr_BMI.npy', allow_pickle=True)
    T1_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/T1_masked_BMI.npy', allow_pickle=True)  # T1_MIMS.npy
    age_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/age_BMI.npy', allow_pickle=True)
    sex_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/sex_BMI.npy', allow_pickle=True)
    id_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/id_BMI.npy', allow_pickle=True)
    MRE_coverage_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/MRE_coverage_BMI.npy', allow_pickle=True)
    study_BMI = np.array(['BMI'] * len(age_BMI))

    stiffness_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/stiffness_NOVA.npy', allow_pickle=True)
    dr_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/dr_NOVA.npy', allow_pickle=True)
    T1_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/T1_masked_NOVA.npy', allow_pickle=True)  # T1_MIMS.npy
    age_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/age_NOVA.npy', allow_pickle=True)
    sex_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/sex_NOVA.npy', allow_pickle=True)
    id_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/id_NOVA.npy', allow_pickle=True)
    MRE_coverage_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/MRE_coverage_NOVA.npy', allow_pickle=True)
    study_NOVA = np.array(['NOVA'] * len(age_NOVA))

    stiffness_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/stiffness_NITRC_batch_1.npy',
                                      allow_pickle=True)
    dr_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/dr_NITRC_batch_1.npy', allow_pickle=True)
    T1_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/T1_masked_NITRC_batch_1.npy',
                               allow_pickle=True)  # T1_NITRC_batch_1.npy
    age_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/age_NITRC_batch_1.npy', allow_pickle=True)
    sex_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/sex_NITRC_batch_1.npy', allow_pickle=True)
    id_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/id_NITRC_batch_1.npy', allow_pickle=True)
    MRE_coverage_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/MRE_coverage_NITRC_batch_1.npy',
                                         allow_pickle=True)
    study_NITRC_batch_1 = np.array(['NITRC_batch_1'] * len(age_NITRC_batch_1))

    stiffness_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/stiffness_NITRC_batch_2.npy',
                                      allow_pickle=True)
    dr_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/dr_NITRC_batch_2.npy', allow_pickle=True)
    T1_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/T1_masked_NITRC_batch_2.npy',
                               allow_pickle=True)  # T1_NITRC_batch_2.npy
    age_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/age_NITRC_batch_2.npy', allow_pickle=True)
    sex_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/sex_NITRC_batch_2.npy', allow_pickle=True)
    id_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/id_NITRC_batch_2.npy', allow_pickle=True)
    MRE_coverage_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/MRE_coverage_NITRC_batch_2.npy',
                                         allow_pickle=True)
    study_NITRC_batch_2 = np.array(['NITRC_batch_2'] * len(age_NITRC_batch_2))

    stiffness_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/stiffness_MIMS.npy', allow_pickle=True)
    dr_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/dr_MIMS.npy', allow_pickle=True)
    T1_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/T1_masked_MIMS.npy', allow_pickle=True)  # T1_MIMS.npy
    age_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/age_MIMS.npy', allow_pickle=True)
    sex_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/sex_MIMS.npy', allow_pickle=True)
    id_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/id_MIMS.npy', allow_pickle=True)
    MRE_coverage_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/MRE_coverage_MIMS.npy', allow_pickle=True)
    study_MIMS = np.array(['MIMS'] * len(age_MIMS))

    if study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
        stiffness_all_healthy = np.concatenate(
            (stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_OA,
             stiffness_MIMS, stiffness_BMI, stiffness_NOVA), axis=0)
        dr_all_healthy = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_OA, dr_MIMS, dr_BMI, dr_NOVA),
                                        axis=0)
        T1_all_healthy = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_OA, T1_MIMS, T1_BMI, T1_NOVA),
                                        axis=0)
        age_all_healthy = np.concatenate(
            (age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_OA, age_MIMS, age_BMI, age_NOVA), axis=0)
        sex_all_healthy = np.concatenate(
            (sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_OA, sex_MIMS, sex_BMI, sex_NOVA), axis=0)
        study_all_healthy = np.concatenate(
            (study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_OA, study_MIMS, study_BMI, study_NOVA),
            axis=0)
        id_all_healthy = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_OA, id_MIMS, id_BMI, id_NOVA),
                                        axis=0)
        MRE_coverage_all_healthy = np.concatenate((MRE_coverage_ATLAS, MRE_coverage_NITRC_batch_1,
                                                   MRE_coverage_NITRC_batch_2, MRE_coverage_OA, MRE_coverage_MIMS,
                                                   MRE_coverage_BMI, MRE_coverage_NOVA), axis=0)

        unique, inverse = np.unique(age_all_healthy, return_inverse=True)
        counts = np.bincount(inverse)
        total_count = age_all_healthy.shape[0]
        imbalance_percentages = counts[inverse] / total_count

        return (
            stiffness_all_healthy, dr_all_healthy, T1_all_healthy, age_all_healthy, sex_all_healthy, study_all_healthy,
            id_all_healthy, imbalance_percentages, MRE_coverage_all_healthy)
