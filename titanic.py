
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

no_of_features = 6;
theta = np.zeros((no_of_features,1));

def eliminate_age_from_train(x):
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

def eliminate_age_from_test(x):
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

    
def sigmoid(z):
    return (1.0/(1.0 + np.exp(-1 * z)));

def cost_function(hyp, y):
    summation = 0.0;
    m = y.shape[0];
    summation = summation + (((y) * (np.log(hyp))) - ((1-y) * (np.log(1-hyp))));
    result = (-1.0/m) * summation;    
    return result;

def theta_grad_calc(X,y,theta,alpha=0.001):
    hyp = sigmoid(np.dot(X,theta));
    m = y.shape[0];
    sub = hyp - y;
    sub = sub.transpose();
    theta_grad = np.dot(sub,X);
    theta_grad = (alpha/m) * theta_grad;
    return theta_grad.transpose();

def threshold_func(x):
    if x > 0.5:
        return 1;
    else:
        return 0;


def train():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df = df_train.groupby('Pclass')['Age'];
    
    df_train['Age'] = df_train[['Age','Pclass']].apply(eliminate_age_from_train,axis=1);
    df_test['Age'] = df_test[['Age','Pclass']].apply(eliminate_age_from_test,axis=1);
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
    y = df_train['Survived'];
    X = df_train.drop('Survived',axis=1)

    X = X.as_matrix(columns=['Pclass','Age','male','SibSp','Parch']);

    
    X_mat = np.asmatrix(X);
    y_mat = np.asmatrix(y);
    
    y_mat = y_mat.reshape((y_mat.shape[1], 1));
    
    #X_norm = (X_mat - np.mean(X_mat,axis=0))/(np.std(X_mat,axis=0))
    X_norm = np.insert(X_mat,0,1,axis=1)
    data = np.concatenate((X_norm,y_mat),axis=1)
    
    global theta;

    for i in range(0,5000):
        X_t = data[...,0:-1];
        y_t = data[...,-1];
        theta = theta - theta_grad_calc(X_t,y_t,theta,0.0005);
        
    print(theta);
    
def test(Pclass, age, male, siblingCount, parchCount):
    X_test = [1,Pclass, age, male, siblingCount, parchCount];
    hyp = np.dot(X_test, theta);
    print(hyp)
    y_test = threshold_func(hyp);
    print(y_test)
    if y_test == 1:
        return "Survived"
    else:
        return "Died"
       
#
## In[86]:
#
#fig = plt.figure()
#axes = fig.add_axes([0,0,1,1]);
#axes.plot(range(0,(500/step_sz)),cost_train,label="Training Cost");
#axes.plot(range(0,(500/step_sz)),cost_cv,label="CV Cost");
#axes.legend();
#
#
## In[87]:
#
#hyp = np.dot(X_norm,theta);
#
#
## In[88]:
#
#pred = np.apply_along_axis(func1d=threshold_func,axis=1,arr=hyp);
#
#
## In[89]:
#
#pred = pred.reshape(pred.shape[0],1)
#
#
## In[90]:
#
#pred.shape
#
#
## In[91]:
#
#from sklearn.metrics import classification_report
#print(classification_report(pred,y_mat))
#
#
## In[92]:
#
#theta
#
#
## In[93]:
#
#df_test = pd.read_csv('titanic_test_data.csv',index_col=0)
#
#
## In[94]:
#
#df_test.info()
#
#
## In[95]:
#
#X_mat = df_test[['Pclass','male','Age','SibSp','Parch']];
#
#
## In[96]:
#
#X_mat.info()
#
#
## In[97]:
#
#X_norm = (X_mat - np.mean(X_mat,axis=0))/(np.std(X_mat,axis=0))
#
#
## In[98]:
#
#X_norm
#
#
## In[99]:
#
#X_norm.insert(loc = 0,column = 'temp', value = 1)
#
#
## In[100]:
#
#X_norm
#
#
## In[101]:
#
#hyp = sigmoid(np.dot(X_norm,theta));
#
#
## In[102]:
#
#hyp.shape
#
#
## In[103]:
#
#pred = np.apply_along_axis(func1d=threshold_func,axis=1,arr=hyp);
#
#
## In[104]:
#
#pred = pred.reshape((418,1))
#
#
## In[105]:
#
#pred
#
#
## In[106]:
#
#pred.shape
#
#
## In[125]:
#
#df_test.dropna(inplace=True,axis=0)
#
#
## In[107]:
#
#submission = pd.DataFrame(pred,columns=['Survived'],index = df_test['PassengerId']);
#
#
## In[131]:
#
#df_test.info(0)
#
#
## In[108]:
#
#submission.shape
#
#
## In[109]:
#
#submission.index = submission.index.map(int)
#
#
## In[ ]:
#
#
#
#
## In[110]:
#
#submission.to_csv('submission.csv')
#
#
## In[ ]:
#
#
#
