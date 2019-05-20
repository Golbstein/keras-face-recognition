import pandas as pd

def get_faces_df(PATH='./'):
    ''' Returns 3 data frames - for train\validation\testing '''
    
    celeb_data = pd.read_csv(PATH + 'identity_CelebA.txt', sep=" ", header=None)
    celeb_data.columns = ["image", "label"]

    # 0 - train, 1 - validation, 2 - test
    train_val_test = pd.read_csv(PATH+'list_eval_partition.csv', usecols=['partition']).values[:, 0]

    df_train = celeb_data.iloc[train_val_test == 0]
    df_valid = celeb_data.iloc[train_val_test == 1]
    df_test = celeb_data.iloc[train_val_test == 2]

    print('Train images:', len(df_train))
    print('Validation images:', len(df_valid))
    print('Test images:', len(df_test))
    
    return df_train, df_valid, df_test, train_val_test
