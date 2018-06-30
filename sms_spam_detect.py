# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 11:31:16 2018

@author: Lenovo
"""
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns





import numpy as np                
import warnings
data = pd.read_csv("spam_ham.csv",encoding='latin-1')
def check_spam_ham(spam_ham_string):
    
    warnings.filterwarnings('ignore')
    
    global data
    
    
    
    #Drop column and name change
    
    data = data.rename(columns={"v1":"label", "v2":"text"})
    #Count observations in each label
    data.label.value_counts()
    
    # convert label to a numerical variable
    data['label_num'] = data.label.map({'ham':0, 'spam':1})
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.2, random_state = 10)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer()
    vect.fit(X_train)
    
    print(vect.get_feature_names()[0:20])
    print(vect.get_feature_names()[-20:])
    
    
    X_train_df = vect.transform(X_train)
    X_test_df = vect.transform(X_test)
    
    
    
    
    
    ham_words = ''
    spam_words = ''
    spam = data[data.label_num == 1]
    ham = data[data.label_num ==0]
    
            
    import nltk
    
    #nltk.download()
    for val in spam.text:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        #tokens = [word for word in tokens if word not in stopwords.words('english')]
        for words in tokens:
            spam_words = spam_words + words + ' '
            
    for val in ham.text:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        for words in tokens:
            ham_words = ham_words + words + ' '
    
   
    
    #naive bayes
    prediction = dict()
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train_df,y_train)
    
    prediction["Multinomial"] = model.predict(X_test_df)
    
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
    accuracy_score(y_test,prediction["Multinomial"])
    
    #logistic regression 
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train_df,y_train)
    
    prediction["Logistic"] = model.predict(X_test_df)
    accuracy_score(y_test,prediction["Logistic"])
    
    #kneighbors classifier
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_df,y_train)
    
    prediction["knn"] = model.predict(X_test_df)
    accuracy_score(y_test,prediction["knn"])
    
    
    #random forest
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train_df,y_train)
    
    prediction["random_forest"] = model.predict(X_test_df)
    accuracy_score(y_test,prediction["random_forest"])
    
    
    from sklearn.model_selection import GridSearchCV
    
    k_range = np.arange(1,30)
    param_grid = dict(n_neighbors=k_range)
    print(param_grid)
    #model = KNeighborsClassifier()
    grid = GridSearchCV(model,param_grid)
    grid.fit(X_train_df,y_train)
    
    grid.best_estimator_
    
    grid.best_params_
    
    grid.best_score_
    
    grid.grid_scores_
    
    print(classification_report(y_test, prediction['Multinomial'], target_names = ["Ham", "Spam"]))
    
    conf_mat = confusion_matrix(y_test, prediction['Multinomial'])
    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_mat_normalized)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    X_test[y_test < prediction["Multinomial"] ]
    X_test[y_test > prediction["Multinomial"] ]
    
    test_str = spam_ham_string
    test_str_series = list()
    test_str_series.append(test_str)
    test_str_df = vect.transform(test_str_series)
    
    predict_result=model.predict(test_str_df)
    
    
    #visualize spam and ham data
    count_Class=pd.value_counts(data["label"], sort= True)
    count_Class.plot(kind= 'bar',color= ["blue", "red"])
    plt.title('Bar chart')
    plt.show()
    return ""

