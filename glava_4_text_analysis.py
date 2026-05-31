import pandas as pd
import matplotlib.pyplot as plt
import re

# 1. Загрузка файла
DATA_PATH = "twcs.csv"

df = pd.read_csv(DATA_PATH)

print(df.shape)
display(df.head())

print("=== 1. Первичное знакомство с данными ===")
print("Файл:", DATA_PATH)
print("Размерность:", df.shape)
print("Столбцы:", df.columns.tolist())

display(df.head())

print("\n=== 2. Пропущенные значения ===")
missing = df.isna().sum()
missing_percent = (df.isna().mean() * 100).round(2)

missing_table = pd.DataFrame({
    "Количество пропусков": missing,
    "Доля пропусков, %": missing_percent
})

display(missing_table)

print("\n=== 3. Распределение сообщений inbound ===")
display(df["inbound"].value_counts())
inbound_table = df["inbound"].value_counts().reset_index()
inbound_table.columns = ["inbound", "count"]
display(inbound_table)
inbound_table.to_csv("inbound_table.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(6, 4))
df["inbound"].value_counts().plot(kind="bar")
plt.title("Распределение сообщений по признаку inbound")
plt.xlabel("Inbound")
plt.ylabel("Количество сообщений")
plt.grid(True)
plt.show()

print("\n=== 4. Анализ длины текстов ===")
df["text"] = df["text"].astype(str)
df["text_len_words"] = df["text"].apply(lambda x: len(x.split()))

display(df["text_len_words"].describe())
text_len_stats = df["text_len_words"].describe().reset_index()
text_len_stats.columns = ["Показатель", "Значение"]
display(text_len_stats)
text_len_stats.to_csv("text_len_stats.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 5))
df["text_len_words"].clip(upper=100).hist(bins=50)
plt.title("Распределение длины сообщений")
plt.xlabel("Количество слов")
plt.ylabel("Количество сообщений")
plt.grid(True)
plt.show()

print("\n=== 5. Очистка текста ===")

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

display(df[["text", "clean_text"]].head())

print("\n=== 6. Формирование пар 'обращение клиента — ответ поддержки' ===")

customers = df[df["inbound"] == True][
    ["tweet_id", "clean_text"]
].copy()

agents = df[df["inbound"] == False][
    ["in_response_to_tweet_id", "clean_text"]
].copy()

pairs = agents.merge(
    customers,
    left_on="in_response_to_tweet_id",
    right_on="tweet_id",
    suffixes=("_answer", "_request")
)

pairs = pairs.rename(columns={
    "clean_text_request": "request",
    "clean_text_answer": "answer"
})

pairs = pairs[["request", "answer"]].dropna().drop_duplicates()

print("Количество пар обращение-ответ:", len(pairs))
display(pairs.head(10))

print("=== Ручная проверка 15 примеров ===")
display(pairs.sample(15, random_state=42))

print("\n=== 7. Анализ длины обращений и ответов ===")

pairs["request_len_words"] = pairs["request"].apply(lambda x: len(x.split()))
pairs["answer_len_words"] = pairs["answer"].apply(lambda x: len(x.split()))

print("Статистика длины обращений:")
display(pairs["request_len_words"].describe())

print("Статистика длины ответов:")
display(pairs["answer_len_words"].describe())

length_stats_table = pd.DataFrame({
    "Показатель": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    "Длина обращений": pairs["request_len_words"].describe().values,
    "Длина ответов": pairs["answer_len_words"].describe().values
})

display(length_stats_table)
length_stats_table.to_csv("request_answer_length_stats.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 5))
pairs["request_len_words"].clip(upper=100).hist(bins=50)
plt.title("Распределение длины обращений клиентов")
plt.xlabel("Количество слов")
plt.ylabel("Количество обращений")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
pairs["answer_len_words"].clip(upper=100).hist(bins=50)
plt.title("Распределение длины ответов службы поддержки")
plt.xlabel("Количество слов")
plt.ylabel("Количество ответов")
plt.grid(True)
plt.show()

print("\n=== 8. Сохранение обработанного датасета ===")

pairs.to_csv("customer_support_pairs.csv", index=False, encoding="utf-8-sig")

print("Файл сохранён: customer_support_pairs.csv")
print("Итоговый формат: request — обращение клиента, answer — ответ поддержки")
