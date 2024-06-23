import pandas as pd


df = pd.read_csv(
    "/home/vinicius/repositories/ppgia/msc/data/output/answers_meta-llama_Meta-Llama-3-70B-Instruct.csv",
    sep=";",
    decimal=","
)


print(df)