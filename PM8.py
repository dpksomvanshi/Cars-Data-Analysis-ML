def catconsep(df):
    cat = list(df.columns[df.dtypes=='object'])
    con = list(df.columns[df.dtypes!='object'])
    return cat,con

def replacer(df):
    cat,con = catconsep(df)
    for i in df.columns:
        if i in cat:
            md = df[i].mode()[0]
            df[i]=df[i].fillna(md)
        else:
            mn = df[i].mean()
            df[i]=df[i].fillna(mn)
    print('Missing Values replaced') 
    
def evaluate_model(xtrain,ytrain,xtest,ytest,model):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ypred_tr = model.predict(xtrain)
    ypred_ts = model.predict(xtest)
    tr_mse = mean_squared_error(ytrain,ypred_tr)
    tr_rmse = tr_mse**(1/2)
    tr_mae = mean_absolute_error(ytrain,ypred_tr)
    tr_r2 = r2_score(ytrain,ypred_tr)
    print('Training Scores :')
    print('MSE : ',tr_mse)
    print('RMSE: ',tr_rmse)
    print('MAE : ',tr_mae)
    print('R2  : ',tr_r2)
    
    print('\n============================\n')
    
    ts_mse = mean_squared_error(ytest,ypred_ts)
    ts_rmse = ts_mse**(1/2)
    ts_mae = mean_absolute_error(ytest,ypred_ts)
    ts_r2 = r2_score(ytest,ypred_ts)
    print('Training Scores :')
    print('MSE : ',ts_mse)
    print('RMSE: ',ts_rmse)
    print('MAE : ',ts_mae)
    print('R2  : ',ts_r2)   

   
    
def r2_adj(xtrain,ytrain,model):
    r = model.score(xtrain,ytrain)
    N = xtrain.shape[0]
    p = xtrain.shape[1]
    num = (1-r)*(N-1)
    den = N-p-1
    adj_r2 = 1-(num/den)
    return adj_r2

def evaluate_model1(xtrain,xtest,ytrain,ytest,model):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ypred_tr = model.predict(xtrain)
    ypred_ts = model.predict(xtest)
    tr_mse = mean_squared_error(ytrain,ypred_tr)
    tr_rmse = tr_mse**(1/2)
    tr_mae = mean_absolute_error(ytrain,ypred_tr)
    tr_r2 = r2_score(ytrain,ypred_tr)
    print('Training Scores :')
    print('MSE : ',tr_mse)
    print('RMSE: ',tr_rmse)
    print('MAE : ',tr_mae)
    print('R2  : ',tr_r2)
    
    print('\n============================\n')
    
    ts_mse = mean_squared_error(ytest,ypred_ts)
    ts_rmse = ts_mse**(1/2)
    ts_mae = mean_absolute_error(ytest,ypred_ts)
    ts_r2 = r2_score(ytest,ypred_ts)
    print('Training Scores :')
    print('MSE : ',ts_mse)
    print('RMSE: ',ts_rmse)
    print('MAE : ',ts_mae)
    print('R2  : ',ts_r2)   