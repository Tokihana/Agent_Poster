from cgi import test
import os
import random
from re import split
import shutil
import argparse

parser = argparse.ArgumentParser() # for shell argument parse
parser.add_argument('-o','--ori', type=str, default='../data/RAF-DB/valid')
parser.add_argument('-a','--aim', type=str, default='../data/RAF-DB-SP')
parser.add_argument('--test_rate', default=0.2)

args = parser.parse_args()

ori_path = args.ori
aim_path = args.aim
rate = args.test_rate

def random_split(names, test_rate = 0.2):
    '''
    return splited name list
    Args:
    names (list): names list
    test_rate (schalar): test_data_ratio
    Returns:
    train_names (list): random splited train_names
    test_names (list): random splited test_names
    '''
    random.shuffle(names)
    split = round(len(names)*(1 - test_rate))
    return names[0:split], names[split:len(names)]

def copy_files(ori, aim, file_names):
    '''
    copy each file in file_names from ori to aim path
    Args:
    ori, aim (str): origin and aim path
    file_names (list): name of files needs to copy
    Return: None
    '''
    if not os.path.exists(aim):
        os.mkdir(aim)
    for file in file_names:
        shutil.copyfile(ori+'/'+file, aim+'/'+file)
        
def split_data(ori_path, aim_path, test_rate = 0.2):
    '''
    split data to train and test split, save to aim path
    Args:
    ori_path, aim_path (str): paths
    test_rate (schalar): rate of test data
    Return: None
    '''
    if not os.path.exists(aim_path):
        os.mkdir(aim_path)
    train_path, test_path = aim_path + '/train', aim_path + '/valid'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    sub_folder = os.listdir(ori_path)
    for sub in sub_folder:
        sub_path = ori_path + '/' + sub
        names = os.listdir(sub_path)
        train_names, test_names = random_split(names)
        # print(train_names, test_names)
        # copy to train
        copy_files(sub_path, train_path + '/' + sub, train_names)
        copy_files(sub_path, test_path + '/' + sub, test_names)
        

if __name__ == '__main__':
    split_data(ori_path, aim_path, test_rate=rate)