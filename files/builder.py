import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def cutting(feature, limite, rpl):
    '''
    For grouping int variable
    rpl = replacement
    limite = bound to cut
    '''
    listing = feature.value_counts().index[feature.value_counts() < limite]
    for it in listing:
        feature.replace(it, rpl, inplace=True)
    return None


def T_application(df):
    '''
    df is the application table
    Dropping, loging, grouping, conversion, features engeniering tasks,
    '''
    #### Deleting features and rows ####
    # Del XNA Gender client
    df = df[df['CODE_GENDER'] != 'XNA']
    #del client with unknown family statut just 2 instantces
    df = df[df['NAME_FAMILY_STATUS'] != 'Unknown'] 
    # del some features
    for n in ['2','4','7','10','12','17','19','20','21']:
        del df['FLAG_DOCUMENT_'+str(n)]
    del df['FLAG_MOBIL']
    del df['FLAG_CONT_MOBILE']
    del df['OCCUPATION_TYPE']
    del df['ORGANIZATION_TYPE'] 
    
    #### Grouping task #####
    # Cuting number of children at 4 and more
#     df.loc[:,'CNT_CHILDREN'] = np.where(df['CNT_CHILDREN']>3, 4, df['CNT_CHILDREN'])
    # Grouping application hour into 3 categories
    df.loc[:,'HOUR_APPR_PROCESS_START'] = np.where(df['HOUR_APPR_PROCESS_START']<8, 1, df['HOUR_APPR_PROCESS_START'])
    df.loc[:,'HOUR_APPR_PROCESS_START'] = np.where((df['HOUR_APPR_PROCESS_START']>1)&(df['HOUR_APPR_PROCESS_START']<18), 2, df['HOUR_APPR_PROCESS_START'])
    df.loc[:,'HOUR_APPR_PROCESS_START'] = np.where(df['HOUR_APPR_PROCESS_START']>2, 3, df['HOUR_APPR_PROCESS_START'])
    
    # Cutting familly size
#     df.loc[:,'CNT_FAM_MEMBERS'] = np.where(df['CNT_FAM_MEMBERS']>6,6, df['CNT_FAM_MEMBERS'])
#     df.loc[:,'CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].astype('Int64')
    cutting(df['NAME_INCOME_TYPE'], 100, 'Other')
    
    #### Conversion ####
    # Convert into positive variable
    0# NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df.loc[:,['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']] = \
    -1 * df[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']]
    
    df.loc[:,['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']] = \
    df[['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']].astype('float')
    
    # Convert into int variable
    for col in ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','CNT_FAM_MEMBERS']:
        df.loc[:,col] = df[col].astype('Int64')
    #loging variable
    for col in ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL',
                'AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR', 'OBS_30_CNT_SOCIAL_CIRCLE',
                'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','OWN_CAR_AGE'] :
        df.loc[:,col] = np.log(1+df[col])
        
    return df

class MultiLabelEncoder:
    """
    For encoding multiple columns
    """
    
    def fit(self, df):
        short = df.select_dtypes(['object'])
        self.col = short.columns.tolist()
        for x in self.col :
            exec(f"self.{x} = {LabelEncoder()}")
            exec(f"self.{x}.fit(df[x])")
        return self
    
    def transform(self, df):
        X = df.copy()
        for x in self.col :
            exec(f"X[x] = self.{x}.transform(X[x])")
        return X
    
    def fit_transform(self, df):
        X = df.copy()
        short = X.select_dtypes(['object'])
        self.col = short.columns.tolist()
        for x in self.col :
            exec(f"self.{x} = {LabelEncoder()}")
            exec(f"X[x] = self.{x}.fit_transform(X[x])")
        return X
    
    def inverse_transform(self, df):
        X = df.copy()
        for x in self.col :
            exec(f"X[x] = self.{x}.inverse_transform(X[x])")
        return X 
    
    def a(self):
        """
        return list of available LabelEncoder
        """
        return self.col
    
def quant_imputer(data,ind_float):
    '''Impute float variable'''
    ## Getting list of float variables + id columns
    float_columns = data.select_dtypes('float').columns.tolist()
    float_columns.append('SK_ID_CURR')
    _df = data[float_columns]
    
    ##Imputation
    imp = SimpleImputer(fill_value=0.55556 ,strategy='constant',add_indicator=ind_float)
    if ind_float==True:
        _cols = _df.columns.tolist() #initial columns list
        x_plus = _df.columns[_df.isna().sum()>0].tolist() #columns with na list

        for i in range(len(x_plus)):
            x_plus[i] = str('NA')+'_'+str(x_plus[i]) #new indiator column nomes

        _df = imp.fit_transform(_df)
        _df = pd.DataFrame(_df)
        _df.columns = _cols + x_plus
        _df[x_plus] = _df[x_plus].astype('int')
        #Mean imputer 0.1464231357858447,   0.15251724829955812 with na indicator
        #Median imputer 0.1481379817484892, 0.1492624942382043 with na indicator
        #0.44445 imputer 0.15095428846998724, 0.15392181283312434 with na indicator
        #0.55556 imputer with na indicator 0.15404084696801165
    else:
        _df.loc[:,:] = imp.fit_transform(_df)
    return _df

def int_imputer(data):
    ''''
    int is imputed with #999 imputer and indicator _missing
    '''
    _df = data.select_dtypes(['int32','int64','Int64'])
    imp = SimpleImputer(fill_value=999, strategy='constant')
#     if ind_int==True:
#         _cols = _df.columns.tolist() #initial columns list
#         x_plus = _df.columns[_df.isna().sum()>0].tolist() #columns with na list

#         for i in range(len(x_plus)):
#             x_plus[i] = str('NA')+'_'+str(x_plus[i]) #new indiator column nomes

#         _df = imp.fit_transform(_df)
#         _df = pd.DataFrame(_df)
#         _df.columns = _cols + x_plus
#         _df[_cols] = _df[_cols].astype('int')
#         _df[x_plus] = _df[x_plus].astype('int')
#     else:
    _df.loc[:,:] = imp.fit_transform(_df)
    _df = _df.astype('int')
    
    # 0         0.15456136738081866, with missing ind 0.1483649964396835
    #most_freq  0.15456136738081866, with ind 0.1483649964396835 
    #99         0.15404084696801165, with 0.15404084696801165
    return _df

def obj_imputer(data):
    obj_columns = data.select_dtypes(['object']).columns.tolist()
    obj_columns.append('SK_ID_CURR')
    _df = data[obj_columns]
    imp = SimpleImputer(fill_value='999', strategy='constant')
    _df.loc[:,:] = imp.fit_transform(_df)
    _df = _df.astype('object')
    return _df


def quant_feature_engineering(df, deleting):
    '''Some  feature tranformations'''
    #Some simple new features (percentages)
    #With new var 0.15347127687237183, +plus deleting old 0.14890910717212885
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_PERSON'] = df['INCOME_PER_PERSON'].astype('float')
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    if deleting==True:
        #deleting
        del df['DAYS_EMPLOYED']
        del df['DAYS_BIRTH']
        del df['AMT_INCOME_TOTAL']
        del df['AMT_CREDIT']
        del df['CNT_FAM_MEMBERS']
        del df['AMT_ANNUITY'] 
    return df

def df_preprocessing(data,ind_float=True, new_var=False, deleting=False):     
    '''
    This function treats int variable before train test split.
    
    ind_float : boolean, whether or not an indicator variable is added when imputation is proceeded for float variables.
    
    ind_int : boolean, whether or not an indicator variable is added when imputation is proceeded for int variables.
    
    dummy: boolean, whether or not variables are converted into dummies
    
    fdrop: boolean, True when first class is deleted when dummies are getting
    
    new_var : boolean, feature_engineering function will be activated? 
        deleting : boolean, is inital variables combined wil be deleted?

    '''
    list_obj = data.select_dtypes('object').columns
    _df =data.copy()
    _df[list_obj] = _df[list_obj].fillna('NA')
    _df = pd.get_dummies(data, drop_first=True, columns=list_obj, dtype='int')
    _df[_df.select_dtypes(['int','Int64']).columns] = _df.select_dtypes(['int','Int64']).fillna(999)
    _df = _df.select_dtypes(exclude='float').merge(quant_imputer(_df,ind_float), how='inner', on='SK_ID_CURR')
    _df[_df.select_dtypes('Int64').columns] = _df[_df.select_dtypes('Int64').columns].astype('int')
    _df[_df.select_dtypes('Float64').columns] = _df[_df.select_dtypes('Float64').columns].astype('float')
    
    try:
        del _df['SK_ID_CURR']
    except:
        None

    #Feature engineering
    if new_var==True:
        try:
            _df = quant_feature_engineering(_df, deleting)   
        except:
            None
        
    return _df

def production_data_fromating(data, col, list_obj, reducer, ind_float=True,new_var=True, deleting=False):
    """
    To set new data as right format
    """
    _df =data.copy()
    _df[list_obj] = _df[list_obj].fillna('NA')
    _df = pd.get_dummies(_df, drop_first=True, columns=list_obj, dtype='int')
    _df[_df.select_dtypes(['int','Int64']).columns] = _df.select_dtypes(['int','Int64']).fillna(999)
    _df = _df.select_dtypes(exclude='float').merge(quant_imputer(_df,ind_float), how='inner', on='SK_ID_CURR')
    _df[_df.select_dtypes('Int64').columns] = _df[_df.select_dtypes('Int64').columns].astype('int')
    _df[_df.select_dtypes('Float64').columns] = _df[_df.select_dtypes('Float64').columns].astype('float')
    
    try:
        del _df['SK_ID_CURR']
    except:
        None

    #Feature engineering
    if new_var==True:
        try:
            _df = quant_feature_engineering(_df, deleting)   
        except:
            None
            
    _df = _df.reindex(columns = col, fill_value=0) #according data features
    
    return _df