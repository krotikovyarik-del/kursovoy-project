import pandas as pd

df = pd.read_csv("Rotating_equipment_fault_data.csv")

print(df.head())
print(df.columns)
DATA_PATH = "Rotating_equipment_fault_data.csv"
print("\nНазвания столбцов:")
print(df.columns)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

DATA_PATH = "Rotating_equipment_fault_data.csv"

# 1. Загрузка данных
df = pd.read_csv(DATA_PATH)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")
df = df.set_index("Timestamp")

print("Первые строки:")
display(df.head())

print("Размерность:", df.shape)
print("Типы данных:")
display(df.dtypes)

# 2. Каналы временного ряда
data_ts = df[["Vibration_X", "Vibration_Y", "Vibration_Z", "Acoustic_Level", "Temperature"]]
label_col = "Fault_Type"

print("Каналы для анализа:")
print(data_ts.columns.tolist())

print("Классы неисправностей:")
display(df[label_col].value_counts())

# 3. Визуализация исходных данных
plt.figure(figsize=(14, 14))

for i, col in enumerate(data_ts.columns, 1):
    plt.subplot(len(data_ts.columns), 1, i)
    plt.plot(data_ts.index, data_ts[col], label=col)
    plt.title(f"Временной ряд: {col}")
    plt.xlabel("Время")
    plt.ylabel(col)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# 4. Температура по классам состояния
plt.figure(figsize=(14, 5))

for fault_type in df[label_col].unique():
    part = df[df[label_col] == fault_type]
    plt.scatter(part.index, part["Temperature"], s=15, label=fault_type)

plt.title("Температура по классам состояния")
plt.xlabel("Время")
plt.ylabel("Temperature")
plt.grid(True)
plt.legend()
plt.show()

# 5. Статистический анализ
stats_table = data_ts.describe().T
stats_table["median"] = data_ts.median()
stats_table["variance"] = data_ts.var()

print("Описательная статистика:")
display(stats_table)

# 6. Частота дискретизации
time_diffs = data_ts.index.to_series().diff().dropna()

print("Интервалы между измерениями:")
display(time_diffs.value_counts().head())

print("Наиболее частый интервал:")
display(time_diffs.mode())

# 7. Анализ пропусков
missing_percent = data_ts.isna().mean() * 100

print("Доля пропущенных значений, %:")
display(missing_percent)

# 8. Выбросы по правилу трех сигм
outlier_counts = {}

for col in data_ts.columns:
    mean = data_ts[col].mean()
    std = data_ts[col].std()

    lower = mean - 3 * std
    upper = mean + 3 * std

    outlier_counts[col] = ((data_ts[col] < lower) | (data_ts[col] > upper)).sum()

outlier_counts = pd.Series(outlier_counts)

print("Количество выбросов по правилу трех сигм:")
display(outlier_counts)

# 9. Диаграммы размаха без нормализации
plt.figure(figsize=(12, 6))
data_ts.boxplot()
plt.title("Диаграммы размаха по каналам без нормализации")
plt.ylabel("Значение")
plt.grid(True)
plt.show()

# 10. Диаграммы размаха после стандартизации
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_ts)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=data_ts.columns,
    index=data_ts.index
)

plt.figure(figsize=(12, 6))
scaled_df.boxplot()
plt.title("Диаграммы размаха по каналам после стандартизации")
plt.ylabel("Стандартизированное значение")
plt.grid(True)
plt.show()

# 11. Анализ диапазонов значений
ranges_table = pd.DataFrame({
    "min": data_ts.min(),
    "max": data_ts.max(),
    "range": data_ts.max() - data_ts.min(),
    "std": data_ts.std()
})

print("Диапазоны значений:")
display(ranges_table)

plt.figure(figsize=(12, 6))
data_ts.plot(kind="box", figsize=(12, 6))
plt.title("Сравнение диапазонов значений всех каналов")
plt.ylabel("Значение")
plt.grid(True)
plt.show()

# 12. Корреляционный анализ
corr_matrix = data_ts.corr(method="pearson")

print("Корреляционная матрица:")
display(corr_matrix)

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar(label="Коэффициент корреляции Пирсона")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Тепловая карта корреляций")
plt.tight_layout()
plt.show()

# 13. Анализ шума и SNR для температуры
# reset_index нужен, чтобы seasonal_decompose работал корректнее при неравномерных датах
series = data_ts["Temperature"].dropna().reset_index(drop=True)

SEASONAL_PERIOD = 24
ZOOM_POINTS = 300

if len(series) >= 2 * SEASONAL_PERIOD:
    decomposition = seasonal_decompose(
        series,
        model="additive",
        period=SEASONAL_PERIOD,
        extrapolate_trend="freq"
    )

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    signal = trend + seasonal

    signal_var = np.nanvar(signal)
    noise_var = np.nanvar(residual)

    snr = 10 * np.log10(signal_var / noise_var)

    print(f"SNR для Temperature = {snr:.2f} дБ")

    if snr > 20:
        print("Оценка: отлично, шум практически незаметен.")
    elif snr > 10:
        print("Оценка: хорошо, шум есть, но сигнал доминирует.")
    elif snr > 0:
        print("Оценка: удовлетворительно, сигнал и шум сравнимы.")
    else:
        print("Оценка: плохо, шум сильнее полезного сигнала.")

    # Увеличенный фрагмент декомпозиции
    plt.figure(figsize=(16, 12))

    plt.subplot(4, 1, 1)
    plt.plot(series.iloc[:ZOOM_POINTS])
    plt.title("Исходный ряд Temperature — увеличенный фрагмент")
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(trend.iloc[:ZOOM_POINTS])
    plt.title("Тренд — увеличенный фрагмент")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(seasonal.iloc[:ZOOM_POINTS])
    plt.title("Сезонная компонента — увеличенный фрагмент")
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(residual.iloc[:ZOOM_POINTS])
    plt.title("Остатки / шум — увеличенный фрагмент")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Отдельно сезонная компонента крупно
    plt.figure(figsize=(16, 4))
    plt.plot(seasonal.iloc[:ZOOM_POINTS])
    plt.title("Сезонная компонента Temperature — увеличенный масштаб")
    plt.xlabel("Номер наблюдения")
    plt.ylabel("Сезонная компонента")
    plt.grid(True)
    plt.show()

    # Гистограмма остатков
    plt.figure(figsize=(10, 5))
    plt.hist(residual.dropna(), bins=40)
    plt.title("Гистограмма распределения остатков Temperature")
    plt.xlabel("Остаток")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.show()

else:
    print("Недостаточно данных для декомпозиции.")

# 14. Сохранение результатов
stats_table.to_csv("statistics_table.csv", encoding="utf-8-sig")
missing_percent.to_csv("missing_percent.csv", encoding="utf-8-sig")
outlier_counts.to_csv("outlier_counts.csv", encoding="utf-8-sig")
ranges_table.to_csv("ranges_table.csv", encoding="utf-8-sig")
corr_matrix.to_csv("correlation_matrix.csv", encoding="utf-8-sig")
scaled_df.to_csv("standardized_data.csv", encoding="utf-8-sig")

print("Готово. Результаты сохранены.")
