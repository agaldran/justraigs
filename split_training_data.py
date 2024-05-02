import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
import os, os.path as osp
from tqdm import tqdm


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
    if label == 'NRG':
        return 0
    elif label == 'RG':
        return 1
    else:
        return -1


def main(args):
    df = pd.read_csv(args.csv_path_in, sep=';')
    df['Final Label'] = df['Final Label'].map(labels)
    df['Label G1'] = df['Label G1'].map(labels)
    df['Label G2'] = df['Label G2'].map(labels)
    df['Label G3'] = df['Label G3'].map(labels)
    df = df.drop([n for n in df.columns if n not in ['Eye ID', 'Final Label', 'Label G1', 'Label G2', 'Label G3']], axis=1)
    df = df.rename(columns={'Eye ID': 'image_id', 'Final Label': 'label'})
    df.image_id = [osp.join('data/images', n + '.JPG') for n in df.image_id.values]

    available_ims = os.listdir('data/images')
    available_ims = [osp.join('data/images', n) for n in available_ims]
    print('before filtering = {}'.format(len(df.image_id)))
    df = df.loc[df.image_id.isin(available_ims)]
    print('after filtering = {}'.format(len(df.image_id)))

    num_ims = len(df)
    meh, df_val1 = train_test_split(df, test_size=num_ims // 5, random_state=0, stratify=df.label)
    meh, df_val2 = train_test_split(meh, test_size=num_ims // 5, random_state=0, stratify=meh.label)
    meh, df_val3 = train_test_split(meh, test_size=num_ims // 5, random_state=0, stratify=meh.label)
    df_val5, df_val4 = train_test_split(meh, test_size=num_ims // 5, random_state=0, stratify=meh.label)

    df_train1 = pd.concat([df_val2, df_val3, df_val4, df_val5], axis=0)
    df_train2 = pd.concat([df_val1, df_val3, df_val4, df_val5], axis=0)
    df_train3 = pd.concat([df_val1, df_val2, df_val4, df_val5], axis=0)
    df_train4 = pd.concat([df_val1, df_val2, df_val3, df_val5], axis=0)
    df_train5 = pd.concat([df_val1, df_val2, df_val3, df_val4], axis=0)

    df_train1.to_csv(osp.join(args.csvs_path_out, 'tr_rg_f1.csv'), index=None)
    df_val1.to_csv(osp.join(args.csvs_path_out, 'vl_rg_f1.csv'), index=None)

    df_train2.to_csv(osp.join(args.csvs_path_out, 'tr_rg_f2.csv'), index=None)
    df_val2.to_csv(osp.join(args.csvs_path_out, 'vl_rg_f2.csv'), index=None)

    df_train3.to_csv(osp.join(args.csvs_path_out, 'tr_rg_f3.csv'), index=None)
    df_val3.to_csv(osp.join(args.csvs_path_out, 'vl_rg_f3.csv'), index=None)

    df_train4.to_csv(osp.join(args.csvs_path_out, 'tr_rg_f4.csv'), index=None)
    df_val4.to_csv(osp.join(args.csvs_path_out, 'vl_rg_f4.csv'), index=None)

    df_train5.to_csv(osp.join(args.csvs_path_out, 'tr_rg_f5.csv'), index=None)
    df_val5.to_csv(osp.join(args.csvs_path_out, 'vl_rg_f5.csv'), index=None)

    df_train_mini = df_train1.sample(frac=0.1, random_state=0)
    df_val_mini = df_val1.sample(frac=0.1, random_state=0)

    im_list = df_train_mini.image_id.values.tolist() + df_val_mini.image_id.values.tolist()
    os.makedirs('data/mini_data/', exist_ok=True)

    for n in tqdm(range(len(im_list))):
        im = im_list[n]
        shutil.copyfile(im, im.replace('data/images', 'data/mini_data'))
    df_train_mini.image_id = [n.replace('data/images/', 'data/mini_data/') for n in df_train_mini.image_id]
    df_val_mini.image_id = [n.replace('data/images/', 'data/mini_data/') for n in df_val_mini.image_id]

    df_train_mini.to_csv('data/tr_mini.csv', index=None)
    df_val_mini.to_csv('data/vl_mini.csv', index=None)


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
