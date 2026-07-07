
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils._repr_html import features
from statsmodels.stats.outliers_influence import variance_inflation_factor
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
sns.set_theme(style="darkgrid")
def load_clean_data(filepath):

    df = pd.read_csv('disney_plus_titles.csv')
    df.loc[df['type'] == 'TV Show', 'director'] = 'No director'
    print(df.head())
    print('='*80)
    print(df.info())
    print('='*80)
    #===========================================
    df['cast_list'] = df['cast'].apply(lambda x: [actor.strip() for actor in str(x).split(',')] if pd.notna(x) else [])

    movie_box_cast = df[(df['type'] == 'Movie') & (df['director'].notna()) & (df['cast'].notna())]
    movie_dontknow_cast = df[(df['type'] == 'Movie') & (df['director'].isna()) & (df['cast'].notna())]



    for idx, row in movie_dontknow_cast.iterrows():
        current_cast = row['cast_list']
        matches = movie_box_cast[movie_box_cast['cast_list'].apply(lambda x: len(set(x) & set(current_cast)) > 0)]

        if not matches.empty:
            df.at[idx, 'director'] = matches['director'].mode()[0]


#==========================================

    movie_box = df[(df['type'] == 'Movie') & (df['director'].notna())]
    movie_dontknow = df[(df['type'] == 'Movie') & (df['director'].isna())]

    for idx, row in movie_dontknow.iterrows():
        current_country = row['country']
        current_genre = row['listed_in']

        if pd.isna(current_country) or pd.isna(current_genre):
            continue

        matches = movie_box[(movie_box['country'] == current_country) & (movie_box['listed_in'] == current_genre)]
        if not matches.empty:
            df.at[idx, 'director'] = matches['director'].mode()[0]


    #===================================
    country_know = df[df['director'].notna() & df['country'].notna()]
    for idx, row in df[df['country'].isna()].iterrows():
        current_dir = row['director']
        matches = country_know[country_know['director'] == current_dir]
        if not matches.empty:
            df.at[idx, 'country'] = matches['country'].mode()[0]
        else:
            df.at[idx, 'country'] = 'United States'


    #=======================================================
    df['director'] = df['director'].fillna('No find dir-r')
    df['cast'] = df['cast'].fillna('No cast')
    df['date_added'] = df['date_added'].fillna('No date')
    df['rating'] = df['rating'].fillna('No rating')
    return df
#================================================
#fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#sns.countplot(data=df, x='type', ax=axes[0], palette='pastel')
#axes[0].set_title('Соотношение Фильмов и Сериалов на Disney')
#axes[0].set_xlabel('Тип контента')
#axes[0].set_ylabel('Количество')

#top_countries = df['country'].value_counts().head(5)
#sns.barplot(x=top_countries.values, y=top_countries.index, ax=axes[1], palette='viridis')
#axes[1].set_title('Топ-5 стран на Disney')
#axes[1].set_xlabel('Количество контента')
#axes[1].set_ylabel('Страна')

#plt.tight_layout()
#plt.show()

# =======================================================
def coding_columns(df):
    df['type'] = df['type'].astype('category')
    df['type_encoded'] = df['type'].cat.codes
    print(df[['type', 'type_encoded']].sample(10))

# =======================================================
    df['date_added_clean'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
    df['year_added'] = df['date_added_clean'].dt.year
    df['month_added'] = df['date_added_clean'].dt.month

    print(df[['date_added', 'year_added', 'month_added']].dropna().head(10))

    #plt.figure(figsize=(10, 5))
    #sns.countplot(data=df, x='year_added', palette='magma')
    #plt.title('В какие месяцы Disney чаще всего добавляет контент?')
    #plt.xlabel('Номер месяца (1 - Январь, 12 - Декабрь)')
    #plt.ylabel('Количество релизов')
    #plt.show()
    #=======================================================
    df['is_festive_season'] = df['month_added'].isin([11, 12]).astype(int)

    df['cast_count'] = df['cast_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return df

#==================================================
def vif_run(df):
    vif_features = ['type_encoded', 'year_added', 'month_added', 'is_festive_season', 'cast_count']

    X = df[vif_features].dropna()

    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data["VIF"]  = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return df
#================================
def eda_regression(df):
    only_movies = df[df['type'] == 'Movie'].copy()
    only_movies = only_movies[only_movies['duration'].notna()]
    only_movies['duration_nums'] = only_movies['duration'].str.replace(' min', '', regex=False).astype(int)
    print(f"Размер выборки фильмов после очистки: {only_movies.shape[0]} строк.")
    print("Базовая статистика по хронометражу (в минутах):")
    print(only_movies['duration_nums'].describe())
    print('-' * 50)
    print(df.head())



    plt.figure(figsize=(8, 5))
    sns.histplot(only_movies['duration_nums'], kde=True, bins=30)
    plt.xlabel('Минуты')
    plt.ylabel('Кол-во')

    plt.figure(figsize=(8, 5))
    top_genres = only_movies['listed_in'].value_counts().head().index
    filter_genres = only_movies[only_movies['listed_in'].isin(top_genres)]
    sns.boxplot(data=filter_genres, x='duration_nums', y='listed_in', palette='Set2')
    plt.xlabel('Минуты')
    plt.ylabel('Жанр')
    plt.tight_layout()
    #plt.show()
    return only_movies
#==============================
def start_regression(only_movies):
    movie_end = pd.get_dummies(only_movies, columns=['listed_in'], drop_first=True)
    y = movie_end['duration_nums']
    drop_cols = [
        'show_id', 'type', 'title', 'director', 'cast', 'country',
        'date_added', 'duration', 'description', 'duration_nums',
        'cast_list', 'date_added_clean', 'rating'
    ]
    X = movie_end.drop(columns=drop_cols, errors='ignore')
    X = X.astype(float)

    print(f"Количество признаков после кодирования жанров: {X.shape[1]}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Обучающая выборка: {X_train.shape[0]} фильмов")
    print(f"Тестовая выборка: {X_test.shape[0]} фильмов")
    print("-" * 50)

    reg_model = RandomForestRegressor(n_estimators=100,random_state=42, criterion='absolute_error')
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)


    print(f"Результаты модели RandomForestRegressor:")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.2f} минут")
    print(f"Среднеквадратичная ошибка (RMSE): {rmse:.2f} минут")
    print(f'r2_score: {r2:.4f}')
    print("-" * 50)
    return reg_model

# ===================================
# def start_learn(df):
#     df['rating'] = df['rating'].astype('category')
#     df['rating_encoded'] = df['rating'].cat.codes
#     y = df['type_encoded']
#     features = ['year_added', 'is_festive_season', 'cast_count', 'rating_encoded']
#
#     ml_df = df[df['year_added'].notna()].copy()
#     X = ml_df[features]
#     y = ml_df.loc[X.index, 'type_encoded']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
#     lr_model = LogisticRegression(max_iter=1000, random_state=42)
#     lr_model.fit(X_train, y_train)
#     lr_pred = lr_model.predict(X_test)
#     lr_acc = accuracy_score(y_test, lr_pred)
#     print('='*80)
#     print('Logistic Regression')
#     print(f'acc = {lr_acc * 100:.2f}%')
#     print(f'Отчет об ошибках')
#     print(classification_report(y_test, lr_pred, target_names=['Movie','TV Show']))
#     print(f'Matrix erorrs')
#     print(confusion_matrix(y_test, lr_pred))
#     print('='*80)
#     print('Random Forest')
#     rf_model = RandomForestClassifier(random_state=42)
#     rf_model.fit(X_train, y_train)
#     rf_pred = rf_model.predict(X_test)
#     rf_acc = accuracy_score(y_test, rf_pred)
#     print(f"acc = {rf_acc * 100:.2f}%")
#     print("Подробный отчет по классам")
#     print(classification_report(y_test, rf_pred, target_names=['Movie', 'TV Show']))
#     print("Матрица ошибок")
#     print(confusion_matrix(y_test, rf_pred))


def main():
    df = load_clean_data('disney_plus_titles.csv')
    df = coding_columns(df)
    only_movies = eda_regression(df)
    reg_model = start_regression(only_movies)

if __name__ == '__main__':
    main()

















































