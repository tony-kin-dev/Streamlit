import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title('EDA: Продажи мороженого (выдуманный датасет)')

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
    data = []
    for city in cities:
        # Для каждого города добавим небольшой сдвиг температуры
        city_temp_shift = np.random.normal(0, 2)
        for i, month in enumerate(months):
            avg_temp = base_temps[i] + city_temp_shift + np.random.normal(0, 1)
            shops = np.random.randint(10, 30)
            # Продажи: зимой низкие, летом высокие, зависят от температуры и числа точек
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
st.write(df.head())
# --- Сделайте скриншот этого блока ---

# Описательные статистики
st.header('2. Описательные статистики')
st.write(df.describe())
# --- Сделайте скриншот этого блока ---

# Распределения признаков
st.header('3. Распределения признаков')
feature = st.selectbox('Выберите числовой признак для гистограммы:', ['Продажи (тыс. шт.)', 'Средняя температура (°C)', 'Количество точек продаж'])
fig, ax = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax)
st.pyplot(fig)
# --- Сделайте скриншот этого блока ---

# Boxplot по городам
st.header('4. Boxplot по городам')
feature2 = st.selectbox('Выберите признак для boxplot:', ['Продажи (тыс. шт.)', 'Средняя температура (°C)', 'Количество точек продаж'], key='box')
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