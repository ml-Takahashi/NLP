import pandas as pd
import math
import pickle

def vectorize_df(df):
    list_df = df.apply(split_text,axis=1)
    tri_gram_df = list_df.apply(tri_gram)
    vector = calc_vector(tri_gram_df)
    return vector

def split_text(text_record):
    return text_record.loc[0].split(" ")

def tri_gram(text_list):
    ans = []
    for text in text_list:
        gram = []
        for i in range(len(text)-2):
            gram.append(text[i:i+3])
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

def main():
    tanka = pd.read_csv("kindai.csv",header=None)
    waka = pd.read_csv("waka_half.csv",header=None)

    waka_vector = vectorize_df(waka)
    tanka_vector = vectorize_df(tanka)

    waka_path = "waka_vector.pkl"
    tanka_path = "tanka_vector.pkl"

    with open(waka_path,"wb") as f1:
        pickle.dump(waka_vector, f1)

    with open(tanka_path,"wb") as f2:
        pickle.dump(tanka_vector, f2)


if __name__=="__main__":
    main()