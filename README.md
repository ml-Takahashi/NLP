# 説明
短歌と和歌をベクトル化してみた

### python vectorization.py　[n]

nで文を分割する文字数を決定。指定しなければn=3で処理される。

短歌と和歌のデータをベクトル化し,vector列とid列で各文のベクトルとidを保持するpd.DataFrameをsave_vector_id_df内に保存。    
各文における単語の出現回数をカウントしたpd.DataFrameをsave_count_id内に保存。  
tanka_count_idとwaka_count_idはvectorization.pyの実行時に生成される途中結果のファイルを格納するためのフォルダ。
