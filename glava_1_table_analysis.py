import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# НАСТРОЙКА ГРАФИКОВ
# ==============================================================================

plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12
})

# ==============================================================================
# 1.1. ЗАГРУЗКА ДАТАСЕТА ИЗ ZIP
# ==============================================================================

possible_zip_names = ['table.zip', 'archive (1).zip', 'archive.zip']

zip_path = None

for name in possible_zip_names:
    if os.path.exists(name):
        zip_path = name
        break

if zip_path is None:
    zip_files = [f for f in os.listdir('/content') if f.lower().endswith('.zip')]
    if len(zip_files) == 0:
        raise FileNotFoundError("В /content не найден zip-архив.")
    zip_path = zip_files[0]

extract_path = '/content/extracted_data'

print("Используется архив:", zip_path)

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    all_files = [
        os.path.join(r, f)
        for r, d, fs in os.walk(extract_path)
        for f in fs
        if f.lower().endswith(('.csv', '.xlsx', '.xls'))
    ]

    if len(all_files) == 0:
        raise FileNotFoundError("В архиве не найден CSV или Excel-файл.")

    print("\nНайденные файлы:")
    for i, file in enumerate(all_files):
        print(i, file)

else:
    raise FileNotFoundError("Архив не найден в /content.")


def read_table(file_path, nrows=None):
    """
    Чтение CSV или Excel.
    """
    if file_path.lower().endswith('.csv'):
        return pd.read_csv(file_path, nrows=nrows, low_memory=False)
    else:
        return pd.read_excel(file_path, nrows=nrows)


# ==============================================================================
# ВЫБОР НУЖНОГО ФАЙЛА СО СТОЛБЦОМ Annual average growth rate
# ==============================================================================

target_col = 'Annual average growth rate'
target_file = None

for file in all_files:
    try:
        temp_df = read_table(file, nrows=5)
        if target_col in temp_df.columns:
            target_file = file
            break
    except Exception as e:
        print(f"Не удалось прочитать файл {file}: {e}")

if target_file is None:
    raise FileNotFoundError(
        "Не найден файл со столбцом 'Annual average growth rate'. "
        "Проверь, есть ли в архиве файл US_PopGR."
    )

print("\nВыбран файл для анализа:")
print(target_file)

df = read_table(target_file)

print("\nПервые строки датасета:")
display(df.head())

print("\nРазмерность исходного датасета:")
print(df.shape)

print("\nНазвания столбцов:")
print(df.columns.tolist())

print("\nТипы данных:")
display(df.dtypes)

# ==============================================================================
# 1.1. ПОДГОТОВКА ОСНОВНЫХ СТОЛБЦОВ
# ==============================================================================

year_col = 'Period'
category_col = 'Economy Label'
economy_col = 'Economy'

required_cols = [target_col, year_col, category_col, economy_col]

for col in required_cols:
    if col not in df.columns:
        raise KeyError(
            f"Не найден столбец: {col}. "
            "Проверь названия столбцов через df.columns.tolist()."
        )

df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
df[economy_col] = pd.to_numeric(df[economy_col], errors='coerce')

# Удаляем служебные footnote-столбцы
cols_to_drop = [c for c in df.columns if 'footnote' in c.lower()]

if len(cols_to_drop) > 0:
    df.drop(columns=cols_to_drop, inplace=True)
    print("\nУдалены служебные столбцы Footnote:")
    print(cols_to_drop)

# ==============================================================================
# 1.2.4. АНАЛИЗ И ОБРАБОТКА ПРОПУСКОВ
# ==============================================================================

df_before_missing = df.copy()

print("\nКоличество пропусков до обработки:")
missing_before = df_before_missing.isnull().sum()
display(missing_before)

# ------------------------------------------------------------------------------
# ГРАФИК 1: ТЕПЛОВАЯ КАРТА ПРОПУСКОВ ДО ОБРАБОТКИ
# ------------------------------------------------------------------------------

plt.figure(figsize=(10, 4))
sns.heatmap(
    df_before_missing.isnull(),
    cbar=False,
    yticklabels=False,
    cmap='mako'
)
plt.title('Тепловая карта пропущенных значений до обработки')
plt.xlabel('Признаки')
plt.ylabel('Наблюдения')
plt.tight_layout()
plt.savefig('missing_heatmap_before.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# ГРАФИК 2: КОЛИЧЕСТВО ПРОПУСКОВ ДО ОБРАБОТКИ
# ------------------------------------------------------------------------------

missing_before_nonzero = missing_before[missing_before > 0]

plt.figure(figsize=(10, 5))
missing_before_nonzero.plot(kind='bar', color='tomato', edgecolor='black')
plt.title('Количество пропущенных значений до обработки')
plt.xlabel('Признак')
plt.ylabel('Количество пропусков')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig('missing_count_before.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# ОБРАБОТКА ПРОПУСКОВ
# ------------------------------------------------------------------------------

df[target_col] = df[target_col].fillna(df[target_col].median())
df[year_col] = df[year_col].fillna(df[year_col].median())
df[economy_col] = df[economy_col].fillna(df[economy_col].median())
df[category_col] = df[category_col].fillna('Unknown_Economy')

# Если есть служебный столбец Missing value — удалим
missing_cols = [c for c in df.columns if 'missing value' in c.lower()]

if len(missing_cols) > 0:
    df.drop(columns=missing_cols, inplace=True)
    print("\nУдалены служебные столбцы Missing value:")
    print(missing_cols)

print("\nКоличество пропусков после обработки:")
missing_after = df.isnull().sum()
display(missing_after)

# ------------------------------------------------------------------------------
# ГРАФИК 3: ТЕПЛОВАЯ КАРТА ПРОПУСКОВ ПОСЛЕ ОБРАБОТКИ
# ------------------------------------------------------------------------------

plt.figure(figsize=(10, 4))
sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap='mako'
)
plt.title('Тепловая карта пропущенных значений после обработки')
plt.xlabel('Признаки')
plt.ylabel('Наблюдения')
plt.tight_layout()
plt.savefig('missing_heatmap_after.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# ГРАФИК 4: СРАВНЕНИЕ ПРОПУСКОВ ДО И ПОСЛЕ ОБРАБОТКИ
# ------------------------------------------------------------------------------

missing_compare = pd.DataFrame({
    'До обработки': missing_before,
    'После обработки': missing_after.reindex(missing_before.index).fillna(0)
})

missing_compare_nonzero = missing_compare[
    (missing_compare['До обработки'] > 0) |
    (missing_compare['После обработки'] > 0)
]

plt.figure(figsize=(10, 5))
missing_compare_nonzero.plot(
    kind='bar',
    figsize=(10, 5),
    edgecolor='black'
)
plt.title('Сравнение количества пропусков до и после обработки')
plt.xlabel('Признак')
plt.ylabel('Количество пропусков')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig('missing_before_after_compare.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# ГРАФИК 5: РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОГО ПРИЗНАКА ДО И ПОСЛЕ ЗАПОЛНЕНИЯ ПРОПУСКОВ
# ------------------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df_before_missing[target_col].dropna(),
    bins=35,
    alpha=0.6,
    label='До обработки пропусков',
    edgecolor='black'
)

plt.hist(
    df[target_col].dropna(),
    bins=35,
    alpha=0.6,
    label='После обработки пропусков',
    edgecolor='black'
)

plt.title('Распределение темпа прироста до и после обработки пропусков')
plt.xlabel(target_col)
plt.ylabel('Частота')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('target_before_after_missing.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ЧИСЛОВЫЕ ПРИЗНАКИ
# ==============================================================================

df['Growth_Rate_Abs'] = df[target_col].abs()
df['Growth_Rate_Squared'] = df[target_col] ** 2
df['Growth_Rate_Log'] = np.log1p(df[target_col] - df[target_col].min() + 1)

df['Period_Normalized'] = (
    df[year_col] - df[year_col].mean()
) / df[year_col].std()

# ==============================================================================
# 1.2.6. УСТРАНЕНИЕ ДУБЛИКАТОВ
# ==============================================================================

print("\nРазмер до удаления дубликатов:")
print(df.shape)

duplicates_count = df.duplicated().sum()
print("Количество дубликатов:", duplicates_count)

df = df.drop_duplicates()

print("Размер после удаления дубликатов:")
print(df.shape)

# ==============================================================================
# 1.2.7. АНАЛИЗ И ОБРАБОТКА ВЫБРОСОВ
# ==============================================================================

df_before_outliers = df.copy()

q_low = df[target_col].quantile(0.01)
q_high = df[target_col].quantile(0.99)

print("\nГраницы клиппинга выбросов:")
print("1% квантиль:", q_low)
print("99% квантиль:", q_high)

outliers_count = (
    (df[target_col] < q_low) |
    (df[target_col] > q_high)
).sum()

print("Количество потенциальных выбросов:", outliers_count)

df[target_col] = df[target_col].clip(lower=q_low, upper=q_high)

# Пересчет производных признаков после клиппинга
df['Growth_Rate_Abs'] = df[target_col].abs()
df['Growth_Rate_Squared'] = df[target_col] ** 2
df['Growth_Rate_Log'] = np.log1p(df[target_col] - df[target_col].min() + 1)

# ------------------------------------------------------------------------------
# ГРАФИК 6: СРАВНЕНИЕ РАСПРЕДЕЛЕНИЯ ДО И ПОСЛЕ ОБРАБОТКИ ВЫБРОСОВ
# ------------------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df_before_outliers[target_col].dropna(),
    bins=35,
    alpha=0.6,
    label='До обработки выбросов',
    edgecolor='black'
)

plt.hist(
    df[target_col].dropna(),
    bins=35,
    alpha=0.6,
    label='После обработки выбросов',
    edgecolor='black'
)

plt.title('Сравнение распределения до и после обработки выбросов')
plt.xlabel(target_col)
plt.ylabel('Частота')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('target_before_after_outliers.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# ГРАФИК 7: BOXPLOT ДО И ПОСЛЕ ОБРАБОТКИ ВЫБРОСОВ
# ------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        df_before_outliers[target_col].dropna(),
        df[target_col].dropna()
    ],
    labels=['До обработки', 'После обработки']
)
plt.title('Boxplot до и после обработки выбросов')
plt.ylabel(target_col)
plt.grid(True)
plt.tight_layout()
plt.savefig('boxplot_before_after_outliers.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# 1.2.1. ДИАГРАММЫ РАСПРЕДЕЛЕНИЯ ЧИСЛОВЫХ ПРИЗНАКОВ
# ==============================================================================

numeric_features = [
    year_col,
    economy_col,
    target_col,
    'Growth_Rate_Abs',
    'Growth_Rate_Squared',
    'Growth_Rate_Log',
    'Period_Normalized'
]

numeric_features = [
    col for col in numeric_features
    if col in df.columns
]

print("\nЧисловые признаки для анализа:")
print(numeric_features)

for col in numeric_features:
    plt.figure(figsize=(9, 5))
    plt.hist(
        df[col].dropna(),
        bins=35,
        color='steelblue',
        edgecolor='black',
        alpha=0.8
    )
    plt.title(f'Распределение признака: {col}', fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Частота')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'hist_{col.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# 1.2.2. ВИЗУАЛИЗАЦИЯ 3 ПАР ПРИЗНАКОВ SEABORN
# ==============================================================================

plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df.sample(min(3000, len(df)), random_state=42),
    x=target_col
)
plt.title('Пара 1: распределение темпа роста населения')
plt.xlabel('Annual average growth rate')
plt.tight_layout()
plt.savefig('pair_1_boxplot_growth_rate.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df.sample(min(3000, len(df)), random_state=42),
    x=year_col,
    y=target_col,
    hue=category_col,
    legend=False
)
plt.title('Пара 2: темп роста населения по периодам')
plt.xlabel('Period')
plt.ylabel('Annual average growth rate')
plt.tight_layout()
plt.savefig('pair_2_scatter_period_growth.png', dpi=300, bbox_inches='tight')
plt.show()

top_economies = df[category_col].value_counts().head(10).index
df_top = df[df[category_col].isin(top_economies)]

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_top,
    x=category_col,
    y=target_col
)
plt.title('Пара 3: темп роста населения по первым 10 экономикам')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pair_3_boxplot_economies.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# 1.2.3. ИНТЕРАКТИВНЫЕ ГРАФИКИ PLOTLY
# ==============================================================================

df_sample = df.sample(n=min(2000, len(df)), random_state=42)

fig_scatter = px.scatter(
    df_sample,
    x=year_col,
    y=target_col,
    color=category_col,
    title='Интерактивный scatter plot: темп роста населения по периодам',
    hover_data=[category_col, economy_col]
)

fig_scatter.show()

trend_df = (
    df.groupby(year_col, as_index=False)[target_col]
    .mean()
)

fig_line = px.line(
    trend_df,
    x=year_col,
    y=target_col,
    title='Интерактивная динамика среднего темпа роста населения'
)

fig_line.show()

# ==============================================================================
# 1.2.5. ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИЙ
# ==============================================================================

corr_matrix = df[numeric_features].corr()

print("\nКорреляционная матрица:")
display(corr_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('Тепловая карта 2: матрица корреляции числовых признаков')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# 1.2.8. УСЛОВНАЯ ФИЛЬТРАЦИЯ
# ==============================================================================

print("\nРезультаты условной фильтрации:")

f1 = df[df[target_col] > df[target_col].median()]
print(f"Фильтр 1: темп роста выше медианы — {f1.shape[0]} строк")

f2 = df[df[year_col] >= df[year_col].median()]
print(f"Фильтр 2: современные периоды — {f2.shape[0]} строк")

f3 = df[
    (df[target_col] > 0) &
    (df[category_col] != 'Individual economies')
]
print(f"Фильтр 3: положительный рост без агрегата Individual economies — {f3.shape[0]} строк")

# ==============================================================================
# 1.2.9. ДОБАВЛЕНИЕ ШУМА В 2 ПРИЗНАКА
# ==============================================================================

np.random.seed(42)

df_before_noise = df.copy()

df['Growth_Rate_Noisy'] = (
    df[target_col] +
    np.random.normal(0, df[target_col].std() * 0.02, len(df))
)

df['Period_Noisy'] = (
    df[year_col] +
    np.random.normal(0, 0.5, len(df))
)

print("\nДобавлен контролируемый гауссовский шум в признаки:")
print("Growth_Rate_Noisy")
print("Period_Noisy")

# ------------------------------------------------------------------------------
# ГРАФИК 8: СРАВНЕНИЕ ТЕМПА РОСТА ДО И ПОСЛЕ ДОБАВЛЕНИЯ ШУМА
# ------------------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df_before_noise[target_col].dropna(),
    bins=35,
    alpha=0.6,
    label='Исходный признак',
    edgecolor='black'
)

plt.hist(
    df['Growth_Rate_Noisy'].dropna(),
    bins=35,
    alpha=0.6,
    label='После добавления шума',
    edgecolor='black'
)

plt.title('Сравнение распределения темпа роста до и после добавления шума')
plt.xlabel(target_col)
plt.ylabel('Частота')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('growth_rate_before_after_noise.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# ГРАФИК 9: SCATTER ДО И ПОСЛЕ ШУМА
# ------------------------------------------------------------------------------

noise_sample = df.sample(min(3000, len(df)), random_state=42)

plt.figure(figsize=(7, 7))
plt.scatter(
    noise_sample[target_col],
    noise_sample['Growth_Rate_Noisy'],
    alpha=0.3
)

min_val = min(noise_sample[target_col].min(), noise_sample['Growth_Rate_Noisy'].min())
max_val = max(noise_sample[target_col].max(), noise_sample['Growth_Rate_Noisy'].max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.title('Сравнение исходного темпа роста и признака с шумом')
plt.xlabel('Исходный темп роста')
plt.ylabel('Темп роста после добавления шума')
plt.grid(True)
plt.tight_layout()
plt.savefig('growth_rate_noise_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# ГРАФИК 10: ПЕРИОД ДО И ПОСЛЕ ДОБАВЛЕНИЯ ШУМА
# ------------------------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df_before_noise[year_col].dropna(),
    bins=35,
    alpha=0.6,
    label='Исходный Period',
    edgecolor='black'
)

plt.hist(
    df['Period_Noisy'].dropna(),
    bins=35,
    alpha=0.6,
    label='Period после добавления шума',
    edgecolor='black'
)

plt.title('Сравнение Period до и после добавления шума')
plt.xlabel('Period')
plt.ylabel('Частота')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('period_before_after_noise.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# 1.2.10. ПРЕОБРАЗОВАНИЕ ЧИСЛОВЫХ ДАННЫХ В КАТЕГОРИАЛЬНЫЕ
# ==============================================================================

df['Growth_Category'] = pd.qcut(
    df[target_col],
    q=3,
    labels=[
        'Низкий темп роста',
        'Средний темп роста',
        'Высокий темп роста'
    ],
    duplicates='drop'
)

print("\nКатегории темпа роста:")
display(df['Growth_Category'].value_counts())

# ==============================================================================
# 1.2.12. ПОВТОРНАЯ ВИЗУАЛИЗАЦИЯ ПОСЛЕ ОБРАБОТКИ
# ==============================================================================

plt.figure(figsize=(9, 5))
plt.hist(
    df[target_col],
    bins=35,
    color='seagreen',
    edgecolor='black',
    alpha=0.8
)
plt.title('Распределение темпа роста после обработки выбросов')
plt.xlabel(target_col)
plt.ylabel('Частота')
plt.grid(True)
plt.tight_layout()
plt.savefig('growth_rate_after_processing.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# 1.2.13. ГРУППИРОВКА
# ==============================================================================

group_table = (
    df.groupby(year_col, as_index=False)[target_col]
    .agg(['mean', 'min', 'max', 'count'])
    .reset_index()
)

print("\nГруппировка по периоду:")
display(group_table.head())

# ==============================================================================
# 1.3.1. ПЕРЕЧИСЛЕНИЕ КАТЕГОРИЙ
# ==============================================================================

print("\nПримеры категорий Economy Label:")
print(df[category_col].unique()[:30])

print("\nКатегории Growth_Category:")
print(df['Growth_Category'].unique())

# ==============================================================================
# 1.3.2. ДИАГРАММЫ КАТЕГОРИАЛЬНЫХ ДАННЫХ
# ==============================================================================

plt.figure(figsize=(10, 5))
df[category_col].value_counts().head(20).plot(kind='bar')
plt.title('Топ-20 категорий Economy Label')
plt.xlabel('Economy Label')
plt.ylabel('Количество')
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig('top_20_economy_label.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x='Growth_Category',
    hue='Growth_Category',
    palette='Set3',
    legend=False
)
plt.title('Распределение категорий темпа роста')
plt.tight_layout()
plt.savefig('growth_category_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# 1.3.3. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ДАННЫХ
# ==============================================================================

le = LabelEncoder()

df['Economy_Label_Encoded'] = le.fit_transform(
    df[category_col].astype(str)
)

print("\nLabel Encoding для Economy Label:")
display(df[[category_col, 'Economy_Label_Encoded']].head())

df = pd.get_dummies(
    df,
    columns=['Growth_Category'],
    prefix='Growth_Category',
    drop_first=False
)

print("\nOne-Hot Encoding для Growth_Category выполнен.")

# ==============================================================================
# 1.3.4. АГРЕГАЦИЯ И РЕДКИЕ КАТЕГОРИИ
# ==============================================================================

category_counts = df[category_col].value_counts()

print("\nКоличество записей по категориям Economy Label:")
display(category_counts.head(20))

rare_categories = category_counts[
    category_counts < 0.01 * len(df)
].index.tolist()

print("\nКоличество редких категорий Economy Label:")
print(len(rare_categories))

# ==============================================================================
# 1.3.5. НОВАЯ СЛОЖНАЯ КАТЕГОРИЯ
# ==============================================================================

df['Demographic_Growth_Zone'] = 'Стабильная зона'

df.loc[
    df[target_col] > df[target_col].quantile(0.75),
    'Demographic_Growth_Zone'
] = 'Зона быстрого роста населения'

df.loc[
    df[target_col] < df[target_col].quantile(0.25),
    'Demographic_Growth_Zone'
] = 'Зона снижения населения'

print("\nНовая категория Demographic_Growth_Zone:")
display(df['Demographic_Growth_Zone'].value_counts())

# ==============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==============================================================================

df.to_csv(
    'processed_un_population_growth_data.csv',
    index=False,
    encoding='utf-8-sig'
)

corr_matrix.to_csv(
    'correlation_matrix.csv',
    encoding='utf-8-sig'
)

missing_compare.to_csv(
    'missing_before_after_compare.csv',
    encoding='utf-8-sig'
)

group_table.to_csv(
    'group_by_period.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\nГотово. Анализ завершён.")
print("Итоговый обработанный датасет сохранён:")
print("processed_un_population_growth_data.csv")
print("\nСозданные графики для презентации:")
print("1. missing_heatmap_before.png")
print("2. missing_count_before.png")
print("3. missing_heatmap_after.png")
print("4. missing_before_after_compare.png")
print("5. target_before_after_missing.png")
print("6. target_before_after_outliers.png")
print("7. boxplot_before_after_outliers.png")
print("8. growth_rate_before_after_noise.png")
print("9. growth_rate_noise_scatter.png")
print("10. period_before_after_noise.png")
print("\nФинальный размер датасета:", df.shape)
