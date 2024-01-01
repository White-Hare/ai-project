import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import confusion_matrix, roc_curve, mean_squared_error 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from data_organizer import clear_data, encode_labels, split_train_and_test


def train(model, x_train, x_test, y_train, y_test):
    model = model.fit(x_train, y_train)
    return model.predict(x_test)



#Train ve Validation Hatasi    
def calc_metrics(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)

    predictions = model.predict(x_train)
    train_error = mean_squared_error(y_train, predictions)#Train Hatasi

    predictions = model.predict(x_test)
    validation_error = mean_squared_error(y_test, predictions)#Validation Hatasi
    
    return train_error, validation_error


def get_metrics(y_actual, y_pred, print_confussion_matrix = False):
    #white, black, draw olmak uzere uc cesit label var
    #confusion matrix 3x3 olacak
    labels = np.unique(y_actual)

    cm = confusion_matrix(y_actual, y_pred, labels = labels)
    fp = cm.sum(axis=0) - np.diag(cm)#false positive
    fn = cm.sum(axis=1) - np.diag(cm)#false negative
    tp = np.diag(cm)#true positive
    tn = cm.sum() - (fp + fn + tp)#true negative

    if(print_confussion_matrix):
        print("----Confusion Matrix----\n")
        print(pd.DataFrame(cm, index = labels, columns = labels))
        print('\n')

    acc = (tp + tn) / (tp + fp + tn + fn)#Dogruluk degerini hesapla
    recall = (tp + tn) / (tp + fp + tn + fn)
    specifity = tn / (tn + fp)
    f_score = 2 * tp / (2 * tp + fp + fn)
        

    fpr, tpr, _ = roc_curve(y_actual, y_pred, pos_label=0)
    fpr1, tpr1, _ = roc_curve(y_actual, y_pred, pos_label=1)
    fpr2, tpr2, _ = roc_curve(y_actual, y_pred, pos_label=2)


    return acc, recall, specifity, f_score, [fpr, fpr1, fpr2], [tpr, tpr1, tpr2]


def add_params(y_test, y_pred, white_params, black_params, draw_params):
    acc, recall, specifity, f_score, fpr, tpr = get_metrics(y_test, y_pred)

    black_params.append((acc[0], recall[0], specifity[0], f_score[0], fpr[0], tpr[0]))
    draw_params.append((acc[1], recall[1], specifity[1], f_score[1], fpr[1], tpr[1]))
    white_params.append((acc[2], recall[2], specifity[2], f_score[2], fpr[2], tpr[2]))


#@st.cache_resource()
def read_and_proccess_data(home_work: int,  iteration_start: int, iteration_end: int):
    iteration_end += 1

    df = pd.read_csv("./dataset/games.csv")
    df = clear_data(df)
    df = encode_labels(df)


    dfc = df.copy()


    white_params, black_params, draw_params = [], [], []
    ks = []
    x_train, x_test, y_train, y_test = split_train_and_test(dfc)


    
    if home_work == 3: #Odev 3
        for k in range(iteration_start, iteration_end):#farkli k degerlerine gore olc
            ks.append(k)
            model = KNeighborsClassifier(n_neighbors=k)

            y_pred = train(model, x_train, x_test, y_train, y_test)
            add_params(y_test, y_pred, white_params, black_params, draw_params)

    if home_work == 4: #Odev 4
        for k in range(iteration_start, iteration_end):
            ks.append(k)
            model = GaussianNB(var_smoothing=k * 0.1) #scilearn icerisinde normal naive bayers bulamadim

            y_pred = train(model, x_train, x_test, y_train, y_test)
            add_params(y_test, y_pred, white_params, black_params, draw_params)

    if home_work == 5: #Odev 5.1
        for k in range(iteration_start, iteration_end):
            ks.append(k)
            model = DecisionTreeClassifier(max_depth=k)

            y_pred = train(model, x_train, x_test, y_train, y_test)
            add_params(y_test, y_pred, white_params, black_params, draw_params)

    if home_work == 6: #Odev 5.2
        for k in range(iteration_start, iteration_end):
            ks.append(k)
            model = DecisionTreeRegressor(max_depth=k) #Veri setimde sayisal degerler de oldugu icin bu algoritmayi kullanamadim

            y_pred = train(model, x_train, x_test, y_train, y_test)
            add_params(y_test, y_pred, white_params, black_params, draw_params)

    if home_work == 7: #Odev 6
        for k in range(iteration_start, iteration_end):
            ks.append(k)
            model = LogisticRegression(max_iter=k)

            y_pred = train(model, x_train, x_test, y_train, y_test)
            add_params(y_test, y_pred, white_params, black_params, draw_params)

    if home_work == 9: #Odev 8
      for k in range(iteration_start, iteration_end):
        ks.append(k)
        model = MLPClassifier(max_iter=k, activation='relu', solver='adam')

        y_pred = train(model, x_train, x_test, y_train, y_test)
        add_params(y_test, y_pred, white_params, black_params, draw_params)

    if home_work == 10: #Odev 9
      for k in range(iteration_start, iteration_end):
        ks.append(k)
        model = SVC(max_iter=k, kernel='poly')

        y_pred = train(model, x_train, x_test, y_train, y_test)
        add_params(y_test, y_pred, white_params, black_params, draw_params)



    b_df = create_data_frame(ks, black_params)
    d_df = create_data_frame(ks, draw_params)
    w_df = create_data_frame(ks, white_params)


    b_roc = create_roc_curve(black_params)
    d_roc = create_roc_curve(draw_params)
    w_roc = create_roc_curve(white_params)

    return (b_df, b_roc), (d_df, d_roc), (w_df, w_roc)
    

def create_data_frame(ks, data):
    #Kolonlarla satirlari degistir
    d = np.array([[d[0], d[1], d[2], d[3]] for d in data])

    d = {"Number Of Iteration": ks, 'Accuracy': d[:, 0] , 'Recall': d[:, 1], 'Specifity': d[:, 2], 'F Score': d[:, 3]}
    return pd.DataFrame(d).set_index('Number Of Iteration')

def create_roc_curve(data):
    fpr, tpr = [], []
    for d in data:
        fpr.append(d[4])
        tpr.append(d[5])
    
    fpr = np.concatenate(fpr)
    tpr = np.concatenate(tpr)

    return pd.DataFrame({
        "False Positive": fpr, 
        "True Positive": tpr,
        "Middle Line": tpr,

    }).set_index('True Positive')