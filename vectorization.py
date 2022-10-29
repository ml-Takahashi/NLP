import pandas as pd
import numpy as np
import math
import argparse
import os
import pickle

def vectorize_df(df,n=3,text_num=1000):
    """
    vector,idを保持するdfと辞書,文ごとの単語の出現回数を保持するdfを取得する関数。
    
    Parameters
    ----------
    df : pd.DataFrame
        和歌or短歌のデータフレーム。
    n : int
        文を区切る文字数。
    text_num : int
        1度に処理する文の数（まとめて処理すると終わらないため）。

    Returns
    -------
    out_df : pd.DataFrame
        vector,idを保持するデータフレーム。
    id_df : pd.DataFrame
        文ごとの単語の出現回数を保持するデータフレーム。
    vec_id_dic : dic
        vector,id,count（出現回数）,wordを保持する辞書。
    """
    list_df = df.apply(split_text,axis=1)
    tri_gram_df = list_df.apply(n_gram,n=n)
    vec_id_dic = calc_vector(tri_gram_df)
    out_df = convert_to_vector(tri_gram_df,vec_id_dic)
    id_df = count_id(out_df,vec_id_dic,start_point=0,text_num=text_num)
    return out_df,id_df,vec_id_dic

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

def calc_vector(gram_df):
    vector = {}
    vector["vector"] = {}  # ベクトルを保持
    vector["id"] = {}      # 単語ごとにidを割り当て
    vector["count"] = {}   # 単語の出現回数をカウント
    vector["word"] = {}    # idからwordを取得する
    r2 = 0
    id = 0
    for line in gram_df:
        for word in line:
            if word in vector["count"].keys():
                vector["count"][word] += 1

            else:
                vector["count"][word] = 1
                vector["id"][word] = id
                vector["word"][id] = word
                id += 1

    for word in vector["count"].keys():
        r2 += vector["count"][word]**2

    r = math.sqrt(r2)
    for word in vector["count"].keys():
        vector["vector"][word] = vector["count"][word] / r

    return vector

def convert_to_vector(df,dic):
    out_vec_list = []
    out_id_list = []
    out_df = pd.DataFrame()
    for i in range(df.shape[0]):
        vec_list = []
        id_list = []
        word_list = df.loc[i]
        for word in word_list:
            vec_list.append(dic["vector"][word])
            id_list.append(dic["id"][word])
        out_vec_list.append(vec_list)
        out_id_list.append(id_list)
    out_df["vector"] = out_vec_list
    out_df["id"] = out_id_list
    return out_df

def count_id(df,dic,start_point=0,text_num=1000):
    i = 0
    id_df = pd.DataFrame(columns=dic["id"])
    height = id_df.shape[0]
    width = id_df.shape[1]
    end_point = start_point + text_num
    for id_list in df["id"][start_point:end_point]:
        id_df.loc[i] = np.zeros(width,dtype=int)
        for id in id_list:
            id_df.loc[i,dic["word"][id]] += 1
        i += 1
        if i==end_point:
            return id_df
    return id_df

def save_df(data_df,n,save_dir,save_file,waka_or_tanka,text_num=1000):
    for i in range((data_df.shape[0]//text_num)+1):
        sp = i*text_num
        save_path = os.path.join(save_dir,f"{i}_"+save_file)
        if sp==0:
            vector_id_df,id_df,vec_id_dic = vectorize_df(data_df,text_num=text_num)
            vector_id_df.to_pickle(f"save_vector_id_df/{n}_gram_{waka_or_tanka}_vector_id.pkl")
        else:
            id_df = count_id(vector_id_df,vec_id_dic,start_point=sp,text_num=text_num)
        id_df.to_pickle(save_path)

def concat_df(data_dir,save_dir,save_file):
    save_name = os.path.join(save_dir,save_file)
    df_list = []
    for file_name in os.listdir(data_dir):
        with open(os.path.join(data_dir,file_name), mode="rb") as f:
            df_list.append(pickle.load(f))
    save_df = pd.concat(df_list,axis=0).reset_index().drop("index",axis=1)
    save_df.to_pickle(save_name)


def main(n):
    waka = pd.read_csv("waka_half.csv",header=None)
    tanka = pd.read_csv("kindai.csv",header=None)

    waka_count_id_path = f"{n}_gram_waka_count_id.pkl"
    tanka_count_id_path = f"{n}_gram_tanka_count_id.pkl"
    waka_save_dir = "waka_count_id"
    tanka_save_dir = "tanka_count_id"
    save_dir = "save_count_id"
    save_df(waka,n,waka_save_dir,waka_count_id_path,waka_or_tanka="waka",text_num=1000)
    save_df(tanka,n,tanka_save_dir,tanka_count_id_path,waka_or_tanka="tanka",text_num=1000)

    concat_df(waka_save_dir,save_dir,waka_count_id_path)
    concat_df(tanka_save_dir,save_dir,tanka_count_id_path)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n",default=[3],nargs="*",type=int,help="文章を分割する文字数を指定")
    args = parser.parse_args()
    main(n=args.n[0])
    