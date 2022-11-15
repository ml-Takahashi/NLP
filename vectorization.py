import pandas as pd
import numpy as np
import math
import argparse
import os
import pickle

def vectorize_df(waka_df,tanka_df,n=3,text_num=1000):
    """
    vector,idを保持するdfと、文ごとの単語の出現回数を保持するdfを取得する関数。
    
    Parameters
    ----------
    df : pd.DataFrame
        和歌or短歌のデータフレーム。
    n : int
        文を区切る文字数。

    Returns
    -------
    {waka or tanka}_count_id_df : pd.DataFrame
        文ごとに出現したidの回数をカウントし、その数を保持するデータフレーム。
    {waka or tanka}_id_list_df : pd.DataFrame
        単語をidに置き換え、そのidの並びを保持するデータフレーム。
    id_dict : dict
        単語ごとにidを割り当て、その対応関係を保持する辞書。
    """
    waka_list_df = waka_df.apply(split_text,axis=1)
    tanka_list_df = tanka_df.apply(split_text,axis=1)
    
    waka_n_gram_df = waka_list_df.apply(n_gram,n=n)
    tanka_n_gram_df = tanka_list_df.apply(n_gram,n=n)

    id_dict = assign_id(pd.concat([waka_n_gram_df,tanka_n_gram_df],axis=0))

    waka_id_list_df = convert_to_id(waka_n_gram_df,id_dict)
    tanka_id_list_df = convert_to_id(tanka_n_gram_df,id_dict)

    waka_count_id_df = count_id(waka_id_list_df,id_dict,start_point=0,text_num=text_num)
    tanka_count_id_df = count_id(tanka_id_list_df,id_dict,start_point=0,text_num=text_num)

    return waka_count_id_df,tanka_count_id_df,waka_id_list_df,tanka_id_list_df,id_dict

def split_text(text_record):
    return text_record.loc[0].split(" ")

def n_gram(text_list,n):
    ans = []
    for text in text_list:
        gram = []
        for i in range(len(text)-(n-1)):
            gram.append(text[i:i+n])
        ans.extend(gram)
    return ans

def assign_id(gram_df):
    id_dict = {}
    id_dict["id"] = {}    # 単語ごとにidを割り当て
    id_dict["word"] = {}  # idからwordを取得する
    id = 0
    for line in gram_df:
        for word in line:
            if word not in id_dict["id"].keys():
                id_dict["id"][word] = id
                id_dict["word"][id] = word
                id += 1
    return id_dict

def count_word(df,dict):
    dict["count"] = {}
    for line in df:
        for word in line:
            if word in dict["count"].keys():
                dict["count"][word] += 1
            else:
                dict["count"][word] = 1
    return 


def convert_to_id(df,dict):
    out_id_list = []
    out_df = pd.DataFrame()
    for i in range(df.shape[0]):
        id_list = []
        word_list = df.loc[i]
        for word in word_list:
            id_list.append(dict["id"][word])
        out_id_list.append(id_list)
    out_df["id"] = out_id_list
    return out_df

def count_id(df,dict,start_point=0,text_num=1000):
    i = 0
    count_id_df = pd.DataFrame(columns=dict["id"])
    width = count_id_df.shape[1]
    end_point = start_point + text_num
    for id_list in df["id"][start_point:end_point]:
        count_id_df.loc[i] = np.zeros(width,dtype=int)
        for id in id_list:
            count_id_df.loc[i,dict["word"][id]] += 1
        i += 1
        if i==end_point:
            return count_id_df
    return count_id_df

def save_df(waka_df,tanka_df,waka_save_dir,tanka_save_dir,waka_save_file,tanka_save_file,text_num=1000):
    for i in range(15):  # 和歌と短歌のデータ数がそれぞれ14292,13963のため
        sp = i*text_num
        waka_save_path = os.path.join(waka_save_dir,f"{i}_"+waka_save_file)
        tanka_save_path = os.path.join(tanka_save_dir,f"{i}_"+tanka_save_file)
        if sp==0:
            (waka_count_id_df,tanka_count_id_df,waka_id_list_df,tanka_id_list_df,
            id_dict) = vectorize_df(waka_df,tanka_df,text_num=text_num)
            waka_id_list_df.to_pickle("id_list_df/waka_id_list_df.pkl")
            tanka_id_list_df.to_pickle("id_list_df/tanka_id_list_df.pkl")
            with open("id_dict.pkl","wb") as f:
                pickle.dump(id_dict,f)
        else:
            waka_count_id_df = count_id(waka_id_list_df,id_dict,start_point=sp,text_num=text_num)
            if i!=14:  # 短歌のデータ数は14000以下のため
                tanka_count_id_df = count_id(tanka_id_list_df,id_dict,start_point=sp,text_num=text_num)
        waka_count_id_df.astype("int32").to_pickle(waka_save_path)
        tanka_count_id_df.astype("int32").to_pickle(tanka_save_path)

def concat_df(data_dir,save_dir,save_file):
    save_name = os.path.join(save_dir,save_file)
    df_list = []
    for file_name in os.listdir(data_dir):
        if file_name[0]==".":  # .DS_Storeを読み込まないようにするため
            continue
        with open(os.path.join(data_dir,file_name), mode="rb") as f:
            df_list.append(pickle.load(f))
    save_df = pd.concat(df_list,axis=0).reset_index().drop("index",axis=1)
    save_df.to_pickle(save_name)


def main(n):
    waka = pd.read_csv("waka_half.csv",header=None)
    tanka = pd.read_csv("kindai.csv",header=None)

    waka_count_id_path = f"{n}_gram_waka_count_id.pkl"
    tanka_count_id_path = f"{n}_gram_tanka_count_id.pkl"
    waka_cp_dir = "waka_checkpoints"
    tanka_cp_dir = "tanka_checkpoints"
    save_dir = "save_count_id_df"
    save_df(waka,tanka,waka_cp_dir,tanka_cp_dir,waka_count_id_path,tanka_count_id_path,text_num=1000)

    concat_df(waka_cp_dir,save_dir,waka_count_id_path)
    concat_df(tanka_cp_dir,save_dir,tanka_count_id_path)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n",default=[3],nargs="*",type=int,help="文章を分割する文字数を指定")
    args = parser.parse_args()
    main(n=args.n[0])