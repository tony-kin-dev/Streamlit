import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title('Продажи мороженого')

st.markdown('''
Это интерактивное приложение для разведочного анализа выдуманного датасета о продажах мороженого в разных городах и месяцах.\n
''')

# Генерация более реалистичных данных
@st.cache_data
def load_fake_data():
    np.random.seed(42)
    cities = ['Москва', 'Санкт-Петербург', 'Казань', 'Новосибирск', 'Екатеринбург']
    months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    # Средние температуры по месяцам (Москва, примерно)
    base_temps = [-8, -7, 0, 8, 17, 21, 24, 23, 16, 8, 1, -5]
    # Базовые значения количества точек продаж по месяцам (зимой меньше, летом больше)
    base_shops = [12, 13, 15, 17, 20, 23, 26, 25, 22, 18, 15, 13]
    data = []
    for city in cities:
        city_temp_shift = np.random.normal(0, 2)
        city_shops_shift = np.random.randint(-2, 3)  # небольшой сдвиг для города
        for i, month in enumerate(months):
            avg_temp = base_temps[i] + city_temp_shift + np.random.normal(0, 1)
            # Плавное изменение количества точек продаж по месяцам
            shops = base_shops[i] + city_shops_shift + np.random.randint(-1, 2)
            shops = max(10, shops)  # не меньше 10
            base_sales = max(0, 10 + (avg_temp if avg_temp > 0 else 0) * 4)
            sales = int(np.random.normal(loc=base_sales + shops * 2, scale=8))
            sales = max(0, sales)
            data.append({
                'Город': city,
                'Месяц': month,
                'Продажи (тыс. шт.)': sales,
                'Средняя температура (°C)': round(avg_temp, 1),
                'Количество точек продаж': shops
            })
    return pd.DataFrame(data)

df = load_fake_data()

# Просмотр данных
st.header('1. Просмотр данных')
selected_city = st.selectbox('Выберите город:', df['Город'].unique())
st.write(df[df['Город'] == selected_city])
# --- Сделайте скриншот этого блока ---

# Описательные статистики
st.header('2. Описательные статистики')
st.write(df.describe())
# --- Сделайте скриншот этого блока ---

# Распределения признаков
st.header('3. Распределения признаков')
feature = st.selectbox('Выберите числовой признак для гистограммы:', ['Продажи (тыс. шт.)', 'Средняя температура (°C)', 'Количество точек продаж'])
fig, ax = plt.subplots()
n_bins = 8
hist = sns.histplot(df[feature], kde=True, ax=ax, bins=n_bins, color='skyblue', edgecolor='black')
# Получаем границы бинов
bins = hist.patches
if bins:
    bin_centers = [patch.get_x() + patch.get_width()/2 for patch in bins]
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f'{x:.0f}' for x in bin_centers])
ax.set_xlabel(feature)
ax.set_ylabel('Count')
st.pyplot(fig)
# --- Сделайте скриншот этого блока ---

# Диаграмма размаха по городам
st.header('4. Диаграмма размаха по городам')
feature2 = st.selectbox('Выберите признак для диаграммы размаха:', ['Продажи (тыс. шт.)', 'Средняя температура (°C)', 'Количество точек продаж'], key='box')
fig2, ax2 = plt.subplots()
sns.boxplot(x='Город', y=feature2, data=df, ax=ax2)
st.pyplot(fig2)
# --- Сделайте скриншот этого блока ---

# Зависимость продаж от температуры
st.header('5. Зависимость продаж от температуры')
fig3, ax3 = plt.subplots()
sns.scatterplot(x='Средняя температура (°C)', y='Продажи (тыс. шт.)', hue='Город', data=df, ax=ax3)
st.pyplot(fig3)
# --- Сделайте скриншот этого блока ---

# Корреляционная матрица
st.header('6. Корреляционная матрица')
corr = df[['Продажи (тыс. шт.)', 'Средняя температура (°C)', 'Количество точек продаж']].corr()
fig4, ax4 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)
# --- Сделайте скриншот этого блока ---