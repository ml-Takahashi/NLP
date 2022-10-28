import pandas as pd
import math
import argparse

def vectorize_df(df,n):
    list_df = df.apply(split_text,axis=1)
    tri_gram_df = list_df.apply(n_gram,n=n)
    vec_id_dic = calc_vector(tri_gram_df)
    out_df = convert_to_vector(tri_gram_df,vec_id_dic)
    return out_df

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
    vector["vector"] = {}
    vector["id"] = {}
    r2 = 0
    id = 0
    for line in gram_df:
        for word in line:
            if word in vector["vector"].keys():
                vector["vector"][word] += 1
            else:
                vector["vector"][word] = 1
                vector["id"][word] = id
            id += 1

    for key in vector["vector"].keys():
        r2 += vector["vector"][key]**2

    r = math.sqrt(r2)
    for key in vector["vector"].keys():
        vector["vector"][key] /= r

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



def main(n):
    tanka = pd.read_csv("kindai.csv",header=None)
    waka = pd.read_csv("waka_half.csv",header=None)

    waka_vector_id = vectorize_df(waka,n)
    tanka_vector_id = vectorize_df(tanka,n)

    waka_path = f"{n}_gram_waka_vector_id.csv"
    tanka_path = f"{n}_gram_tanka_vector_id.csv"

    waka_vector_id.to_csv(waka_path,index=False)
    tanka_vector_id.to_csv(tanka_path,index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n",default=[3],nargs="*",type=int,help="文章を分割する文字数を指定")
    args = parser.parse_args()
    main(n=args.n[0])