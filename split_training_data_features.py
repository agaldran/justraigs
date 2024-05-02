from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os, os.path as osp
from tqdm import tqdm
import shutil
pd.set_option('mode.chained_assignment',None)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Split Data')
    parser.add_argument('--csv_path_in', type=str, default='data/JustRAIGS_Train_labels.csv',
                        help='path to training csvs')
    parser.add_argument('--csvs_path_out', type=str, default='data/', help='path to store k-fold csvs')
    # parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=True, help='avoid saving anything')

    args = parser.parse_args()

    return args

def labels(label):
    if label=='NRG': return 0
    elif label=='RG': return 1
    elif label=='U': return 2
    elif label==np.nan: return 3

def build_labels_g1g3(df, col):
    if df['G1 '+col]==df['G3 '+col]: return df['G3 '+col]
    else:
        if df['G3 '+col]==0: return 0.1
        elif df['G3 '+col]==1: return 0.9
        else: print('wtf')

def build_labels_g2g3(df, col):
    if df['G2 '+col]==df['G3 '+col]: return df['G3 '+col]
    else:
        if df['G3 '+col]==0: return 0.1
        elif df['G3 '+col]==1: return 0.9
        else: print('wtf')

def build_labels_g1g2(df, col):
    if df['G1 '+col]==df['G2 '+col]: return df['G1 '+col]
    else: return 0.5

def build_labels_g1(df, col):
    if df['G1 '+col]==1: return 0.25
    else: return 0

def build_labels_g2(df, col):
    if df['G2 '+col]==1: return 0.25
    else: return 0

def main(args):
    df = pd.read_csv(args.csv_path_in, sep=';')
    df = df.rename(columns={'Eye ID': 'image_id'})
    df.image_id = [osp.join('data/images', n + '.JPG') for n in df.image_id.values]
    available_ims = os.listdir('data/images')
    available_ims = [osp.join('data/images', n) for n in available_ims]
    print('before filtering = {}'.format(len(df.image_id)))
    df = df.loc[df.image_id.isin(available_ims)]
    print('after filtering = {}'.format(len(df.image_id)))

    df['Final Label'] = df['Final Label'].map(labels)
    df['Label G1'] = df['Label G1'].map(labels)
    df['Label G2'] = df['Label G2'].map(labels)
    df['Label G3'] = df['Label G3'].map(labels)

    # only G3 gave opinion
    df_G3_alone = df[(df['Label G3'] == 1) & (df['Label G1'] != 1) & (df['Label G2'] != 1)]
    df_G3_alone = df_G3_alone.drop([n for n in df.columns if 'G1' in n], axis=1)
    df_G3_alone = df_G3_alone.drop([n for n in df.columns if 'G2' in n], axis=1)
    df_G3_alone = df_G3_alone.drop(['Label G3', 'Final Label', 'Fellow Eye ID', 'Age'], axis=1)
    df_G3_alone = df_G3_alone.rename(columns=lambda x: x.replace('G3 ', '') if 'G3' in x else x)
    df_G3_alone = df_G3_alone[['image_id', 'ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']]

    # G1 thought Glaucoma, G3 agreed, G2 disagreed
    df_G1G3 = df[(df['Label G3'] == 1) & (df['Label G1'] == 1) & (df['Label G2'] != 1)]
    df_G1G3['ANRS'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'ANRS'), axis=1)
    df_G1G3['ANRI'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'ANRI'), axis=1)

    df_G1G3['RNFLDS'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'RNFLDS'), axis=1)
    df_G1G3['RNFLDI'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'RNFLDI'), axis=1)

    df_G1G3['BCLVS'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'BCLVS'), axis=1)
    df_G1G3['BCLVI'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'BCLVI'), axis=1)

    df_G1G3['NVT'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'NVT'), axis=1)
    df_G1G3['DH'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'DH'), axis=1)

    df_G1G3['LD'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'LD'), axis=1)
    df_G1G3['LC'] = df_G1G3.apply(lambda x: build_labels_g1g3(x, 'LC'), axis=1)

    df_G1G3 = df_G1G3[['image_id', 'ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']]

    # G2 thought Glaucoma, G3 agreed, G1 disagreed
    df_G2G3 = df[(df['Label G3'] == 1) & (df['Label G2'] == 1) & (df['Label G1'] != 1)] # 382
    df_G2G3['ANRS'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'ANRS'), axis=1)
    df_G2G3['ANRI'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'ANRI'), axis=1)

    df_G2G3['RNFLDS'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'RNFLDS'), axis=1)
    df_G2G3['RNFLDI'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'RNFLDI'), axis=1)

    df_G2G3['BCLVS'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'BCLVS'), axis=1)
    df_G2G3['BCLVI'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'BCLVI'), axis=1)

    df_G2G3['NVT'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'NVT'), axis=1)
    df_G2G3['DH'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'DH'), axis=1)

    df_G2G3['LD'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'LD'), axis=1)
    df_G2G3['LC'] = df_G2G3.apply(lambda x: build_labels_g2g3(x, 'LC'), axis=1)

    df_G2G3 = df_G2G3[['image_id', 'ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']]

    # G1 thought Glaucoma, G2 agreed, but maybe not at feature level
    df_G1G2 = df[(df['Label G1'] == 1) & (df['Label G2'] == 1)]
    df_G1G2['ANRS'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'ANRS'), axis=1)
    df_G1G2['ANRI'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'ANRI'), axis=1)

    df_G1G2['RNFLDS'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'RNFLDS'), axis=1)
    df_G1G2['RNFLDI'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'RNFLDI'), axis=1)

    df_G1G2['BCLVS'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'BCLVS'), axis=1)
    df_G1G2['BCLVI'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'BCLVI'), axis=1)

    df_G1G2['NVT'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'NVT'), axis=1)
    df_G1G2['DH'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'DH'), axis=1)

    df_G1G2['LD'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'LD'), axis=1)
    df_G1G2['LC'] = df_G1G2.apply(lambda x: build_labels_g1g2(x, 'LC'), axis=1)

    df_G1G2 = df_G1G2[['image_id', 'ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']]

    # G1 thought Glaucoma, G2 and G3 disagreed
    df_G1 = df[(df['Label G1'] == 1) & (df['Label G2'] == 0) & (df['Label G3'] == 0)]

    df_G1['ANRS'] = df_G1.apply(lambda x: build_labels_g1(x, 'ANRS'), axis=1)
    df_G1['ANRI'] = df_G1.apply(lambda x: build_labels_g1(x, 'ANRI'), axis=1)

    df_G1['RNFLDS'] = df_G1.apply(lambda x: build_labels_g1(x, 'RNFLDS'), axis=1)
    df_G1['RNFLDI'] = df_G1.apply(lambda x: build_labels_g1(x, 'RNFLDI'), axis=1)

    df_G1['BCLVS'] = df_G1.apply(lambda x: build_labels_g1(x, 'BCLVS'), axis=1)
    df_G1['BCLVI'] = df_G1.apply(lambda x: build_labels_g1(x, 'BCLVI'), axis=1)

    df_G1['NVT'] = df_G1.apply(lambda x: build_labels_g1(x, 'NVT'), axis=1)
    df_G1['DH'] = df_G1.apply(lambda x: build_labels_g1(x, 'DH'), axis=1)

    df_G1['LD'] = df_G1.apply(lambda x: build_labels_g1(x, 'LD'), axis=1)
    df_G1['LC'] = df_G1.apply(lambda x: build_labels_g1(x, 'LC'), axis=1)

    df_G1 = df_G1[['image_id', 'ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']]

    # G2 thought Glaucoma, G1 and G3 disagreed
    df_G2 = df[(df['Label G2'] == 1) & (df['Label G1'] == 0) & (df['Label G3'] == 0)]
    df_G2['ANRS'] = df_G2.apply(lambda x: build_labels_g2(x, 'ANRS'), axis=1)
    df_G2['ANRI'] = df_G2.apply(lambda x: build_labels_g2(x, 'ANRI'), axis=1)

    df_G2['RNFLDS'] = df_G2.apply(lambda x: build_labels_g2(x, 'RNFLDS'), axis=1)
    df_G2['RNFLDI'] = df_G2.apply(lambda x: build_labels_g2(x, 'RNFLDI'), axis=1)

    df_G2['BCLVS'] = df_G2.apply(lambda x: build_labels_g2(x, 'BCLVS'), axis=1)
    df_G2['BCLVI'] = df_G2.apply(lambda x: build_labels_g2(x, 'BCLVI'), axis=1)

    df_G2['NVT'] = df_G2.apply(lambda x: build_labels_g2(x, 'NVT'), axis=1)
    df_G2['DH'] = df_G2.apply(lambda x: build_labels_g2(x, 'DH'), axis=1)

    df_G2['LD'] = df_G2.apply(lambda x: build_labels_g2(x, 'LD'), axis=1)
    df_G2['LC'] = df_G2.apply(lambda x: build_labels_g2(x, 'LC'), axis=1)

    df_G2 = df_G2[['image_id', 'ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']]

    ####################################################################
    # First option (mad): use only reliable annotations (although soft):
    df_features_mad = pd.concat([df_G3_alone, df_G1G3, df_G2G3, df_G1G2], axis=0)

    num_ims = len(df_features_mad)
    meh, df_val1 = train_test_split(df_features_mad, test_size=num_ims // 5, random_state=0)
    meh, df_val2 = train_test_split(meh, test_size=num_ims // 5, random_state=0)
    meh, df_val3 = train_test_split(meh, test_size=num_ims // 5, random_state=0)
    df_val5, df_val4 = train_test_split(meh, test_size=num_ims // 5, random_state=0)

    df_train1 = pd.concat([df_val2, df_val3, df_val4, df_val5], axis=0)
    df_train2 = pd.concat([df_val1, df_val3, df_val4, df_val5], axis=0)
    df_train3 = pd.concat([df_val1, df_val2, df_val4, df_val5], axis=0)
    df_train4 = pd.concat([df_val1, df_val2, df_val3, df_val5], axis=0)
    df_train5 = pd.concat([df_val1, df_val2, df_val3, df_val4], axis=0)

    df_train1.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_mad_f1.csv'), index=None)
    df_val1.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_mad_f1.csv'), index=None)

    df_train2.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_mad_f2.csv'), index=None)
    df_val2.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_mad_f2.csv'), index=None)

    df_train3.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_mad_f3.csv'), index=None)
    df_val3.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_mad_f3.csv'), index=None)

    df_train4.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_mad_f4.csv'), index=None)
    df_val4.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_mad_f4.csv'), index=None)

    df_train5.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_mad_f5.csv'), index=None)
    df_val5.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_mad_f5.csv'), index=None)

    ####################################################################
    # Second option (maddder): add unreliable annotations to training sets:
    df_features_madder = pd.concat([df_G1, df_G2], axis=0)

    df_train1 = pd.concat([df_train1, df_features_madder], axis=0)
    df_train2 = pd.concat([df_train2, df_features_madder], axis=0)
    df_train3 = pd.concat([df_train3, df_features_madder], axis=0)
    df_train4 = pd.concat([df_train4, df_features_madder], axis=0)
    df_train5 = pd.concat([df_train5, df_features_madder], axis=0)
    df_train1.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_madder_f1.csv'), index=None)
    df_val1.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_madder_f1.csv'), index=None)

    df_train2.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_madder_f2.csv'), index=None)
    df_val2.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_madder_f2.csv'), index=None)

    df_train3.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_madder_f3.csv'), index=None)
    df_val3.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_madder_f3.csv'), index=None)

    df_train4.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_madder_f4.csv'), index=None)
    df_val4.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_madder_f4.csv'), index=None)

    df_train5.to_csv(osp.join(args.csvs_path_out, 'tr_features_soft_madder_f5.csv'), index=None)
    df_val5.to_csv(osp.join(args.csvs_path_out, 'vl_features_soft_madder_f5.csv'), index=None)


    df_train_mini = df_train1.sample(frac=0.1, random_state=0)
    df_val_mini = df_val1.sample(frac=0.1, random_state=0)

    im_list = df_train_mini.image_id.values.tolist() + df_val_mini.image_id.values.tolist()
    os.makedirs('data/mini_data/', exist_ok=True)

    for n in tqdm(range(len(im_list))):
        im = im_list[n]
        shutil.copyfile(im, im.replace('data/images', 'data/mini_data'))
    df_train_mini.image_id = [n.replace('data/images/', 'data/mini_data/') for n in df_train_mini.image_id]
    df_val_mini.image_id = [n.replace('data/images/', 'data/mini_data/') for n in df_val_mini.image_id]

    df_train_mini.to_csv('data/tr_mini_madder.csv', index=None)
    df_val_mini.to_csv('data/vl_mini_madder.csv', index=None)

    ####################################################################
    # Third option: sanity. Hard labels. Just keep final label, if there is G3 we keep it, else, we do random selection
    def build_labels_g2overrules(df, col):
        return df['G2 ' + col]

    df_G1G2 = df[(df['Label G1'] == 1) & (df['Label G2'] == 1)]
    df_G1G2['ANRS'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'ANRS'), axis=1)
    df_G1G2['ANRI'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'ANRI'), axis=1)

    df_G1G2['RNFLDS'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'RNFLDS'), axis=1)
    df_G1G2['RNFLDI'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'RNFLDI'), axis=1)

    df_G1G2['BCLVS'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'BCLVS'), axis=1)
    df_G1G2['BCLVI'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'BCLVI'), axis=1)

    df_G1G2['NVT'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'NVT'), axis=1)
    df_G1G2['DH'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'DH'), axis=1)

    df_G1G2['LD'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'LD'), axis=1)
    df_G1G2['LC'] = df_G1G2.apply(lambda x: build_labels_g2overrules(x, 'LC'), axis=1)

    df_G1G2 = df_G1G2.drop([n for n in df.columns if 'G1' in n], axis=1)
    df_G1G2 = df_G1G2.drop([n for n in df.columns if 'G2' in n], axis=1)
    df_G1G2 = df_G1G2.drop([n for n in df.columns if 'G3' in n], axis=1)
    df_G1G2 = df_G1G2.drop(['Final Label', 'Fellow Eye ID', 'Age'], axis=1)

    df_features_sane = pd.concat([df_G3_alone, df_G1G2], axis=0)

    num_ims = len(df_features_sane)
    meh, df_val1 = train_test_split(df_features_sane, test_size=num_ims // 5, random_state=0)
    meh, df_val2 = train_test_split(meh, test_size=num_ims // 5, random_state=0)
    meh, df_val3 = train_test_split(meh, test_size=num_ims // 5, random_state=0)
    df_val5, df_val4 = train_test_split(meh, test_size=num_ims // 5, random_state=0)

    df_train1 = pd.concat([df_val2, df_val3, df_val4, df_val5], axis=0)
    df_train2 = pd.concat([df_val1, df_val3, df_val4, df_val5], axis=0)
    df_train3 = pd.concat([df_val1, df_val2, df_val4, df_val5], axis=0)
    df_train4 = pd.concat([df_val1, df_val2, df_val3, df_val5], axis=0)
    df_train5 = pd.concat([df_val1, df_val2, df_val3, df_val4], axis=0)

    df_train1.to_csv(osp.join(args.csvs_path_out, 'tr_features_sane_f1.csv'), index=None)
    df_val1.to_csv(osp.join(args.csvs_path_out, 'vl_features_sane_f1.csv'), index=None)

    df_train2.to_csv(osp.join(args.csvs_path_out, 'tr_features_sane_f2.csv'), index=None)
    df_val2.to_csv(osp.join(args.csvs_path_out, 'vl_features_sane_f2.csv'), index=None)

    df_train3.to_csv(osp.join(args.csvs_path_out, 'tr_features_sane_f3.csv'), index=None)
    df_val3.to_csv(osp.join(args.csvs_path_out, 'vl_features_sane_f3.csv'), index=None)

    df_train4.to_csv(osp.join(args.csvs_path_out, 'tr_features_sane_f4.csv'), index=None)
    df_val4.to_csv(osp.join(args.csvs_path_out, 'vl_features_sane_f4.csv'), index=None)

    df_train5.to_csv(osp.join(args.csvs_path_out, 'tr_features_sane_f5.csv'), index=None)
    df_val5.to_csv(osp.join(args.csvs_path_out, 'vl_features_sane_f5.csv'), index=None)




if __name__ == "__main__":
    args = get_args_parser()
    main(args)
