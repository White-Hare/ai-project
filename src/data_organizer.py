from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Veriyi temizle ve filitrele
def clear_data(df: DataFrame, sample_size = 0, number_of_most_played_games = 200):
    #Veriyi Temizle ve Duzenle

    if sample_size > 0:
        df = df.sample(n = sample_size)# Test ederken boyutunu kucult

    df = df[['id', 'opening_name', 'white_rating', 'black_rating', 'winner']]#kullanacagin kolonlari al
    df = df.set_index('id')


    most_played_games = df['opening_name'].value_counts().sort_values().tail(number_of_most_played_games).keys()
    df = df.loc[df['opening_name'].isin(most_played_games)]#Cok az ornegi olan satirlari ele.

    df['opening_name'] = df['opening_name'].str.strip().str.replace(r'(:.\|)|(:.)', '', regex = True)#Acilista onemsiz varyasyonlarini gormezden gel. Regular Expression kullanarak filitreledim
    df['opening_name'] = df['opening_name'].str.wrap(15)#Okunurlugu artir. Yoksa grafiklerde labellar tasiyor.



    #df = df[df['rated'] == True]
    df = df[abs(df['white_rating'] - df['black_rating']) < 100]#Elo'lar arasindaki farkin cok olmasi istenilen bir durum degil

    lowestElo, highestElo = df['white_rating'].min(), df['white_rating'].max() 

    lowestElo = int(lowestElo / 100) * 100
    highestElo = int(highestElo / 100) * 100 + 100

     #Elo araliklari icin yeni kolon olusturdum.
    df['rating_range'] = np.nan

    for elo in range(lowestElo, highestElo, 100):
        df['rating_range'] = np.where(df['white_rating'].between(elo, elo + 100), f'{elo}-{elo + 100}', df['rating_range'])

    #print(df['rating_range'].unique())


    return df


#Label'lara karsilik numerik deger ata
def encode_labels(df):
    labelencoder = LabelEncoder()

    dfc = df.copy()
    dfc['opening_name'] = labelencoder.fit_transform(df['opening_name'])
    dfc['rating_range'] = labelencoder.fit_transform(df['rating_range'])
    dfc['winner'] = labelencoder.fit_transform(df['winner'])



    return dfc


def split_train_and_test(df, test_size = 0.25):#Dorte birini test verisi olarak al
    x = df[['opening_name', 'white_rating', 'black_rating', 'rating_range']]
    y = df['winner']
    return train_test_split(x, y, test_size = test_size, random_state = 42) #Veriyi test ve train olarak ikiye ayir.
