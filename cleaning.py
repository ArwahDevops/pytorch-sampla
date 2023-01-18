import pandas as pd

#membaca file csv
df = pd.read_csv("data.csv")

#menyimpan jumlah baris sebelum drop
before_drop = len(df)

#filter baris yang sesuai kriteria dan menyimpan ke dalam variable df
df = df[df["Result"] >= 1000]

#menyimpan jumlah baris sesudah drop
after_drop = len(df)

#menyimpan data yang sesuai kriteria ke dalam file csv baru
df.to_csv("filtered_data.csv", index=False)

#mencetak jumlah baris sebelum dan sesudah drop
print(f"Jumlah baris sebelum drop : {before_drop}")
print(f"Jumlah baris sesudah drop : {after_drop}")
