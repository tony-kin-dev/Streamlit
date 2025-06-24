import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Заголовок приложения
st.title('EDA на датасете Iris')

# Описание
st.markdown('''
Это интерактивное приложение для разведочного анализа данных (EDA) на классическом датасете **Iris**. 
Вы можете изучить распределения признаков, взаимосвязи между ними и основные статистики.
''')

# Загрузка данных
@st.cache_data
def load_data():
    return sns.load_dataset('iris')

df = load_data()

# Просмотр данных
st.header('1. Просмотр данных')
st.write(df.head())

# Скриншот: таблица с данными
# --- Сделайте скриншот этого блока ---

# Основные статистики
st.header('2. Описательные статистики')
st.write(df.describe())

# Скриншот: описательные статистики
# --- Сделайте скриншот этого блока ---

# Распределения признаков
st.header('3. Распределения признаков')
feature = st.selectbox('Выберите признак для гистограммы:', df.columns[:-1])
fig, ax = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax)
st.pyplot(fig)

# Скриншот: гистограмма
# --- Сделайте скриншот этого блока ---

# Boxplot по видам
st.header('4. Boxplot по видам ириса')
feature2 = st.selectbox('Выберите признак для boxplot:', df.columns[:-1], key='box')
fig2, ax2 = plt.subplots()
sns.boxplot(x='species', y=feature2, data=df, ax=ax2)
st.pyplot(fig2)

# Скриншот: boxplot
# --- Сделайте скриншот этого блока ---

# Парные графики
st.header('5. Парные графики (Pairplot)')
if st.checkbox('Показать pairplot (может быть долго)'):
    fig3 = sns.pairplot(df, hue='species')
    st.pyplot(fig3)

# Скриншот: pairplot
# --- Сделайте скриншот этого блока ---

# Корреляционная матрица
st.header('6. Корреляционная матрица')
corr = df.corr(numeric_only=True)
fig4, ax4 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

# Скриншот: корреляционная матрица
# --- Сделайте скриншот этого блока ---

st.markdown('''---\n
**Как развернуть:**\n
1. Зарегистрируйтесь на [Streamlit Cloud](https://streamlit.io/cloud).\n2. Загрузите этот проект на GitHub.\n3. Создайте новое приложение, укажите путь к файлу `streamlit/app.py`.\n4. Готово!\n''') 