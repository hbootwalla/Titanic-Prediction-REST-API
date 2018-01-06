
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

class Predictor:
    def __init__(self):
        self.no_of_features = 6;
        self.epochs = 5000;
        self.alpha = 0.0005;
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')        
        df_train['Age'] = df_train[['Age','Pclass']].apply(self.eliminate_age_from_train,axis=1);
        df_test['Age'] = df_test[['Age','Pclass']].apply(self.eliminate_age_from_test,axis=1);
        df_train.dropna(inplace=True);
        df_test.dropna(inplace=True);
        
        sex = pd.get_dummies(df_train['Sex'],drop_first=True)
        emb = pd.get_dummies(df_train['Embarked'],drop_first=True)
        df_train = pd.concat([df_train,sex],axis=1)
        df_train = pd.concat([df_train,emb],axis=1)
        
        sex = pd.get_dummies(df_test['Sex'],drop_first=True)
        emb = pd.get_dummies(df_test['Embarked'],drop_first=True)
        df_test = pd.concat([df_test,sex],axis=1)
        df_test = pd.concat([df_test,emb],axis=1)
    
        
        df_train.to_csv('titanic_train_data.csv')
        df_test.to_csv('titanic_test_data.csv')
        self.y = df_train['Survived'];
        X = df_train.drop('Survived',axis=1)
    
        self.X = X.as_matrix(columns=['Pclass','Age','male','SibSp','Parch']);
        
    def set_learning_rate(self, alpha):
        self.alpha = alpha;
        
    def set_epochs(self, epochs):
        self.epochs = epochs;
    
    def eliminate_age_from_train(self, x):
        age = x[0];
        pclass = x[1];
        
        if(pd.isnull(age)):
            if(pclass == 1):
                return 38;
            elif(pclass == 2):
                return 30;
            elif(pclass == 3):
                return 25;
        else:
            return age;
    
    def eliminate_age_from_test(self, x):
        age = x[0];
        pclass = x[1];
        
        if(pd.isnull(age)):
            if(pclass == 1):
                return 41;
            elif(pclass == 2):
                return 29;
            elif(pclass == 3):
                return 24;
        else:
            return age;
    
        
    def sigmoid(self, z):
        return (1.0/(1.0 + np.exp(-1 * z)));
    
    def cost_function(self, hyp, y):
        summation = 0.0;
        m = y.shape[0];
        summation = summation + (((y) * (np.log(hyp))) - ((1-y) * (np.log(1-hyp))));
        result = (-1.0/m) * summation;    
        return result;
    
    def theta_grad_calc(self, X,y,theta,alpha=0.001):
        hyp =self.sigmoid(np.dot(X,theta));
        m = y.shape[0];
        sub = hyp - y;
        sub = sub.transpose();
        theta_grad = np.dot(sub,X);
        theta_grad = (alpha/m) * theta_grad;
        return theta_grad.transpose();
    
    def threshold_func(self, x):
        if x > 0.5:
            return 1;
        else:
            return 0;
    
    
    def train(self):
        
        self.theta = np.zeros((self.no_of_features,1));
        
        X_mat = np.asmatrix(self.X);
        y_mat = np.asmatrix(self.y);
        
        y_mat = y_mat.reshape((y_mat.shape[1], 1));
        
        #X_norm = (X_mat - np.mean(X_mat,axis=0))/(np.std(X_mat,axis=0))
        X_norm = np.insert(X_mat,0,1,axis=1)
        data = np.concatenate((X_norm,y_mat),axis=1)
        
        
    
        for i in range(0,self.epochs):
            X_t = data[...,0:-1];
            y_t = data[...,-1];
            self.theta = self.theta - self.theta_grad_calc(X_t,y_t,self.theta,self.alpha);
            
        print(self.theta);
        
    def test(self, Pclass, age, male, siblingCount, parchCount):
        X_test = [1,Pclass, age, male, siblingCount, parchCount];
        hyp = np.dot(X_test, self.theta);
        print(hyp)
        y_test = self.threshold_func(hyp);
        print(y_test)
        if y_test == 1:
            return "Survived"
        else:
            return "Died"
      
