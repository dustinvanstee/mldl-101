# lc_utils.py

# Code functions that are needed to run this lab
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import math

import pandas as pd
#from pandas import scatter_matrix
from pandas.plotting import scatter_matrix


#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import glob

# custom library for some helper functions 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
import seaborn as sns
from itertools import compress
import itertools
import operator
import myenv as myenv

print("CLASS_ENVIRONMENT = {}".format(myenv.CLASS_ENVIRONMENT))
if(myenv.CLASS_ENVIRONMENT == 'dv-mac') :
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras import regularizers
    from keras.models import load_model
elif(myenv.CLASS_ENVIRONMENT == 'nimbix') :
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from tensorflow.keras.models import load_model
elif(myenv.CLASS_ENVIRONMENT == 'acc') :
    import tensorflow as tf
    from tensorflow.python.keras.layers import Input, Dense
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras import regularizers
    from tensorflow.python.keras.models import load_model
else :
    print("ERROR loading myenv.py")


# utility print function
def nprint(mystring) :
    print("**{}** : {}".format(sys._getframe(1).f_code.co_name,mystring))

def load_sample_data(location='/data/work/osa/2018-04-lendingclub/lending-club-loan-data/lendingclub.com/') :
    #For lab force LoanStats_securev1_2018Q1.csv
    loanstats_csv_files = None
    if(myenv.CLASS_ENVIRONMENT == 'nimbix') :
        location='/dl-labs/mldl-101/lab5-powerai-lc/'
        nprint("Setting data location to {}".format(location))
        loanstats_csv_files = glob.glob(location + 'LoanStats_securev1_2016Q1*csv.gz')  # 'LoanStats_secure*csv'
    elif(myenv.CLASS_ENVIRONMENT == 'acc') :
        location='./'
        nprint("Setting data location to {}".format(location))
        loanstats_csv_files = glob.glob(location + 'LoanStats_securev1_2016Q1*csv.gz')  # 'LoanStats_secure*csv'
    else :
        loanstats_csv_files = glob.glob(location + 'LoanStats_securev1_2016Q1*csv')  # 'LoanStats_secure*csv'
    loan_list = []
    nprint("CSV files = {}".format(loanstats_csv_files))
    for i in range(1) : #len(loanstats_csv_files)
        nprint("Loading {}".format(loanstats_csv_files[i]))
        loan_list.append( pd.read_csv(loanstats_csv_files[i], index_col=None, header=1))
        loan_df = pd.concat(loan_list,axis=0)
    return loan_df

def quick_overview_1d(loan_df) :
    nprint("There are " + str(len(loan_df)) + " observations in the dataset.")
    nprint("There are " + str(len(loan_df.columns)) + " variables in the dataset.")

    nprint("\nCategorical vs Numerical")
    dt = pd.DataFrame(loan_df.dtypes)
    dt.columns = ['type']
    nprint("use df.dtypes ...")
    print(dt['type'].value_counts())



    nprint("\n******************Dataset Descriptive Statistics (numerical columns only) *****************************\n")
    print(" running df.describe() ....")
    return loan_df.describe()

def quick_overview_2d(loan_df, cols) :
    nprint("There are " + str(len(loan_df)) + " observations in the dataset.")
    nprint("There are " + str(len(loan_df.columns)) + " variables in the dataset.")

    df = loan_df[cols]
    corr_df = df.corr()
    # plot the heatmap
    sns.set_style(style = 'white')
    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr_df, cmap=cmap,
        xticklabels=corr_df.columns,
        yticklabels=corr_df.columns,vmin=-1.0,vmax=1.0)


def create_loan_default(df) :
    # use a lamba function to encode multiple loan_status entries into a single 1/0 default variable
    nprint("Unique values in loan_status")
    print(df['loan_status'].value_counts())

    df['default'] = df['loan_status'].isin([
        'Default',
        'Charged Off',
        'Late (31-120 days)',
        'Late (16-30 days)',
        'Does not meet the credit policy. Status:Charged Off'
    ]).map(lambda x: int(x))
    
    # Now that we converted loan_status, drop it for later predictions using just default column
    nprint("Dropping other values that are highly correlated with loan_status")
    nprint("Dropping loan_status,total_rec_prncp,total_pymnt,total_pymnt_inv")


    df = df.drop(['loan_status', 'total_rec_prncp','total_pymnt','total_pymnt_inv'], axis=1)
    nprint("Unique values in default")
    print(df['default'].value_counts())

    return df

def clean_lendingclub_data(df) :
    nprint(" Running a couple routines to clean the data ...")
    df = drop_sparse_numeric_columns(df, threshold=0.025)
    nprint("Current DF shape = {}".format(df.shape))
    df = drop_columns(df)
    nprint("Current DF shape = {}".format(df.shape))
    df = impute_columns(df)
    nprint("Current DF shape = {}".format(df.shape))
    df = handle_employee_length(df)
    nprint("Current DF shape = {}".format(df.shape))
    df = handle_revol_util(df)
    nprint("Current DF shape = {}".format(df.shape))
    df = drop_rows(df)
    nprint("Current DF shape = {}".format(df.shape))
    return df


# This function is only useful for numeric columns .  Essentially, run a describe, and
# remove any amount of columns that have values <= a sparsity threshold
def drop_sparse_numeric_columns(df, threshold=0.01) :
    nprint("Dropping columns with less than {} pct cells populated".format(threshold))
    class useless_columns(BaseEstimator, TransformerMixin):
        def __init__(self) :
            a=0
        def fit(self,X,y=None) :
            return self # do nothing, no implementation
        def transform(self,X,y=None) :
            assert isinstance(X, pd.DataFrame)
            remove_cols = []
            # Use describe to filter out columns with a lot junk ...
            X_desc = X.describe()
            count_idx = 0
            max_rows = max([X_desc[i][count_idx] for i in X_desc.columns])

            for c in X_desc.columns :
                #count is the 0th index
                col_sparsity_val = float(X_desc[c][count_idx]) / float(max_rows)

                if( col_sparsity_val <= threshold) : 
                    nprint("Dropping {} since its {} pct populated".format(c,col_sparsity_val))
                    remove_cols.append(c)
                    #print(remove_cols)
            X = X.drop(columns=remove_cols, axis=1)
            return X


    uc = useless_columns()
    df = uc.transform(df)
    return df


def drop_columns(df) :
    nprint("Dropping columns based on lack of examples ..")

    nprint("Initial number of columns = {}".format(len(df.columns)))

    # Note that in the output of describe, I have some columns with less than my 39999 rows.. this is due to NaN 
    # loan_short_df = loan_short_df.fillna(0)
    # loan_short_df[loan_short_df.isnull().any(axis=1)].shape
    # Print out rows with NaNs --> loan_short_df[loan_short_df.isnull().any(axis=1)].head()
    
    drop_list = ['url','debt_settlement_flag_date','next_pymnt_d']
    drop_dates = ['payment_plan_start_date','last_pymnt_d','last_credit_pull_d']
    
    drop_nlp_cand = ['title','emp_title','desc']
    # create hardship indicator -> drop hardship ....
    drop_hardship =['hardship_end_date','hardship_flag','hardship_loan_status','hardship_reason','hardship_start_date','hardship_status','hardship_type']
    
    #create settlement indictor -> 
    drop_settles = [ 'settlement_date','settlement_status']
    
    # Handle NaN for months since ..... these numbers should be high ..
    #<- transform logic
    drop_msince = ['mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl']
    
    # These could be imputed, but drop for now ...
    drop_total = ['total_rev_hi_lim','tot_coll_amt','tot_cur_bal','tot_hi_cred_lim','total_il_high_credit_limit']
    
    # there is information here .. imputer later (maybe based on GRADE ?)
    drop_nums = ['num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m']
    drop_joint = ['verification_status_joint']

    df = df.drop(columns=drop_list,axis=1).\
            drop(columns=drop_dates,axis=1).\
            drop(columns=drop_nlp_cand,axis=1).\
            drop(columns=drop_hardship,axis=1).\
            drop(columns=drop_settles,axis=1).\
            drop(columns=drop_msince,axis=1).\
            drop(columns=drop_total,axis=1).\
            drop(columns=drop_nums,axis=1).\
            drop(columns=drop_joint,axis=1)

    nprint("Final number of columns = {}".format(len(df.columns)))

    return df

def impute_columns(df) :
    nprint("Imputing Values for some columns that are mostly populated")

    # Fill 0 candidates for now - Add justification later ...
    df['percent_bc_gt_75'].fillna(0,inplace=True)
    df['bc_open_to_buy'].fillna(0,inplace=True)
    df['bc_util'].fillna(0,inplace=True)
    df['pct_tl_nvr_dlq'].fillna(0,inplace=True)  # Percent trades never delinquent
    df['avg_cur_bal'].fillna(3000,inplace=True)  # set to around lower 25% percentile
    df['acc_open_past_24mths'].fillna(4,inplace=True)  # set to around lower 25% percentile
    df['mort_acc'].fillna(1,inplace=True)  # set to around lower 25% percentile
    
    df['total_bal_ex_mort'].fillna(18000,inplace=True)# set to around lower 50% percentile
    df['total_bc_limit'].fillna(7800,inplace=True)# set to around lower 50% percentile
    return df


# create emp_length indictor variable 
# emp_length <- impute that with simple formula based on diff mdl ...
def handle_employee_length(df) :
    nprint("Binning employee length into 3 categories")
    def emp_func(row):
        if(isinstance(row['emp_length'], str)) :
            if row['emp_length'] == '1 years' or row['emp_length'] == '2 years' or row['emp_length'] == '3 years':
                return '0_3yrs'
            elif row['emp_length'] == '4 years' or row['emp_length'] == '5 years' or row['emp_length'] == '6 years':
                return '4_6yrs' 
            else:
                return 'gt_6yrs'
        else :
            return '0_3yrs'
    
    df['emp_bin'] = df.apply(emp_func, axis=1)
    df.drop(columns='emp_length',axis=1,inplace=True)
    return df

def handle_revol_util(df) :
    nprint("Cleaning missing values for revol_util")

    def revol_util_func(row) :
        if(isinstance(row['revol_util'], int)) :
            return row['revol_util']
        else :
            return float(row['revol_bal']/(row['revol_bal']+row['loan_amnt']))
    
    df['revol_util_1'] = df.apply(revol_util_func, axis=1)
    df.drop(columns='revol_util',axis=1,inplace=True)
    return df


def columns_with_nans(df) :
    nprint(df.isna().sum())

# dropping rows with lots of NaNs
def drop_rows(df) :
    r0 = len(df)
    df = df.dropna()
    r1 = len(df)

    nprint("Removed {} rows due to high number of NaN".format(r0-r1))
    return df


def create_time_features(df) :
    nprint("Creating new column called time_history : Calculated feature showing how long applicant has been a borrower..")
    class clean_time_columns(BaseEstimator, TransformerMixin):
        def __init__(self) :
            a=0
        def fit(self,X,y=None) :
            return self # do nothing, no implementation
        def transform(self,X,y=None) :
            assert isinstance(X, pd.DataFrame)
            X = X.copy()
            # turn MM-YYYY into YYYY-MM-DD
            X['issue_d'] = X['issue_d'].map(lambda x: datetime.strptime(x, "%b-%Y"))
            X['earliest_cr_line'] = X['earliest_cr_line'].map(lambda x: datetime.strptime(x, '%b-%Y'))
            X['time_history'] = X['issue_d'] - X['earliest_cr_line']
            X['time_history'] = X['time_history'].astype('timedelta64[D]').astype(int)
    
            return X

    cln = clean_time_columns()
    df = cln.transform(df)
    return df


# One-hot encoder for all categorical varaibles
# If cardinality < 50, will build, otherwise drop for now ....
def one_hot_encode_keep_cols(df) :
    cardinatity_limit = 50 
    # This first section identifies the columns to keep and drop based on cardinality
    cat_df = df.select_dtypes(include=['object'])

    cat_df = cat_df.apply(pd.Series.nunique).reset_index()
    cat_df = cat_df.rename(columns={'index' : 'column_name', 0:'count'})
    keep_cols = cat_df[(cat_df['count'] < cardinatity_limit) & (cat_df['count'] >1)]
    drop_cols = cat_df[(cat_df['count'] >= cardinatity_limit) | (cat_df['count'] == 1)]

    cat_keep_list = list(keep_cols['column_name'].values)
    cat_drop_list = list(drop_cols['column_name'].values)
    nprint("Dropping these columns since they are greater than cardinality limit of {}".format(cardinatity_limit))
    nprint(cat_drop_list)
    nprint("Keeping these cols")
    nprint(cat_keep_list)
    # This second  section performs the one hot encoding on the columns identified in cat_keep_list

    tmp_dummies_df_list = [df]
    for cat in cat_keep_list :
        tmp_dummies_df_list.append(pd.get_dummies(df[cat]))
        
        
    df2 = pd.concat(tmp_dummies_df_list, axis=1)
    df2 = df2.drop(columns=cat_keep_list)
    df2 = df2.drop(columns=cat_drop_list)
    df2['id'] = df['id']
    #.drop(cat_keep_list)
    return df2

def plot_histograms(df) :
    for c in df.columns :
        if(df[c].dtype == 'float64' or df[c].dtype == 'int64') :
            plt.figure(c)
            df[c].hist(bins=50, figsize=(10,2)).set_xlabel(c)
        else :
            print("skipping column {} of type  {}".format(c,df[c].dtype))

def bob_heatmap_lc(df,sortColumn, add_corr=False):
    df = df.drop(['id','index'], axis=1)
    df = df.sample(n=1000)
    # scale it all on a 0-1 range ....
    df = (df - df.min(0)) / (df.max(0) -df.min(0))

    plt.figure(figsize=(30, 20))
    
    CMAP = "gist_heat"
    #CMAP = "gist_ncar"
    
    if(sortColumn not in df.columns): # check if a sort column was passed in
        ax = sns.heatmap(df, yticklabels = 1000, cmap=CMAP, center = 1)
    else:
        ax = sns.heatmap(df.sort_values(by=[sortColumn]), yticklabels = 1000, cmap=CMAP, center = 1)

    if(add_corr) :
        corr_vs_1var(df, sortColumn) 

def corr_vs_1var(df, var_to_corr) :
    corr_dict = {}

    nprint("removing object types")
    df = df.select_dtypes(exclude=['object'])

    for c in df.columns :
        if(c != var_to_corr and c != 'index') :
            #nprint("{} {}".format(c,type(c)))
            corr_dict[c] = df[var_to_corr].corr(df[c])
    
    sorted_list = sorted(corr_dict.items(), key=operator.itemgetter(1))
    print("{:60s} {:20s}".format("Highest Positive", "Highest Negative"))
    print("{:60s} {:20s}".format("Correlation", "Correlation"))
    print("{:60s} {:20s}".format("==============", "=============="))
    for x in range(20) :
        print("{:60s} {:20s}".format(str(sorted_list[x]), str(sorted_list[-1-x])))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap,aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig('confusion_mat.png', bbox_inches='tight', format='png', dpi=300, pad_inches=0,transparent=True)
    plt.show()
    return


################################### Machine Learning Funcs ############################################

class lendingclub_ml:


    def __init__(self,df):
        self.df = df
        self.train_df = None
        self.test_df = None

        self.X_train_scaled = None 
        self.Y_train = None 
        self.X_test_scaled = None 
        self.Y_test = None 
        self.pca_model = None
        self.ae_model = None

    
    def create_train_test(self, test_size=0.4) :
        # Train / Test split
        nprint("Dropping 2 timestamp columns issue_d and earliest_cr_line")
        self.df = self.df.drop(['issue_d','earliest_cr_line'],1)
    
        self.train_df, self.test_df = train_test_split(self.df, test_size=test_size, random_state=52)
        
        X_train = self.train_df.drop(['default','id'],1)
        self.Y_train = self.train_df['default']
        
        X_test = self.test_df.drop(['default','id'],1)
        self.Y_test = self.test_df['default']
        # Normalize ??
        
        # Scale the Data Here in a common way
        scaler = StandardScaler(with_mean=True,with_std=True)
        X_train_scaled = scaler.fit_transform(X_train)   # same as (df-df.mean())/df.std(ddof=0)
        X_test_scaled = scaler.transform(X_test)   # 
        
        self.X_train_scaled = pd.DataFrame(data=X_train_scaled,columns=X_train.columns)
        self.X_test_scaled  = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)
    
        nprint("Training set size: " + str(self.train_df.shape))
        nprint("Testing set size: " + str(self.test_df.shape))
        nprint("Train set loan_default:loan_paid ratio : " + str(self.train_df[self.train_df.default == 1].shape[0]) +'/' + str(self.train_df[self.train_df.default != 1].shape[0]))
        nprint("Test  set loan_default:loan_paid ratio : " + str(self.test_df[self.test_df.default == 1].shape[0])+'/' + str(self.test_df[self.test_df.default != 1].shape[0]))
    
    
    def build_pca_model(self,n_components=20) :
        # display sorted eigenvalues
    
        # start w n_components principal components and see how much variance that gives me
        # create instance of PCA (constructor)
        pca = PCA(n_components=n_components)
        # pca = PCA(copy=True, iterated_power='auto', n_components=n_components, random_state=None, svd_solver='auto', tol=0.0, whiten=False) 
        pca.fit(self.X_train_scaled)
        nprint("Explained Variance : {}".format(pca.explained_variance_ratio_))
        self.pca_model = pca
        # Plot the values on a scree plot
        self.pca_scree(n_components)



    
    def pca_scree(self, n_components=6) :
        
        # bin is my x axis variable
        bin = []
        for i in range (int(n_components)):
            bin.append(i+1)
        # plot the cummulative variance against the index of PCA
        cum_var = np.cumsum(self.pca_model.explained_variance_ratio_)
        plt.plot(bin, cum_var)
        # plot the 95% threshold, so we can read off count of principal components that matter
        plt.plot(bin, [.95]*n_components, '--')
        plt.plot(bin, [.75]*n_components, '--')
        plt.plot(bin, [.50]*n_components, '--')
        #turn on grid to make graph reading easier
        plt.grid(True)
        #plt.rcParams.update({'font.size': 24})
        plt.suptitle('PC Variance Explained')
        plt.xlabel('Number of PC Components', fontsize=18)
        plt.ylabel('Fraction of Variance \nExplained', fontsize=16)
        # control number of tick marks, 
        plt.xticks([i for i in range(0,n_components)])
        plt.show()
    
    def build_ae_model(self, ae_layers=[25,5,25], regularization=0.001, folds=2, epochs=10, batch_size=32, k_tries=1) :
    #TODO : add experiment for batch size
    
        r = regularization
        kf = KFold(n_splits=folds)
    
        nprint("Dataframe shape = {}".format(self.X_train_scaled.shape))
        for idx, (train_index, cv_index) in enumerate(kf.split(self.X_train_scaled)):
            nprint("idx={}, xlen={} trainlen={} cvlen={}".format(idx, len(self.X_train_scaled), len(train_index), len(cv_index)))
        
        # Get number of columns in training data
        num_cols = len(self.X_train_scaled.columns) 
        
        # Build Autoencoder based on specification
        nprint("Building Autoencoder using this definition : {} {} {} ".format(num_cols, str(ae_layers),num_cols))
        input_layer = Input(shape=(num_cols, ), name="input_layer")
        layer = []
        layer.append(input_layer)
    
        for l in range(len(ae_layers)) :
            layer.append (Dense(ae_layers[l], activation='relu', kernel_regularizer=regularizers.l2(r), name="dense_"+str(l))(layer[l]) )
        
        # see 'Stacked Auto-Encoders' in paper
        decoded = Dense(num_cols, activation= 'linear', kernel_regularizer=regularizers.l2(r), name="final_layer")(layer[-1]) 
        
        
        # construct and compile AE model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
        
        # train autoencoder - Using Kfolds!
        recon_kfold_loss = 0
        for my_try in range(k_tries) :
            for kf_iter, (train_index, cv_index) in enumerate(kf.split(self.X_train_scaled)):
        
                nprint("Inner Loop encoding_dim : {}, regular={}, kfold_iter={}".format(ae_layers,r,kf_iter))
                X_train, X_CV = self.X_train_scaled.iloc[train_index,:], self.X_train_scaled.iloc[cv_index,:]
                
                nprint("X_train shape={} X_CV shape={}".format(X_train.shape,X_CV.shape))
                
                fit_err = autoencoder.fit(X_train, X_train, shuffle=False, epochs=epochs, batch_size = batch_size,verbose=True, validation_data=(X_CV,X_CV))
                loss = fit_err.history['loss'][-1]
            
                # Sum Loss over all folds
                # L1 implemenation
                #recon_kfold_err =  np.linalg.norm(autoencoder.predict(X_CV)-X_CV)
                #recon_error = recon_error + recon_kfold_err
                # L2 implemenation (mse)
                recon_kfold_loss +=  loss
                #recon_error = recon_error + recon_kfold_err
                nprint("Recon err = {} : Current Iter err = {}".format(recon_kfold_loss,loss) )
        
    
        print(autoencoder.summary())
        # Average error over kfolds ...
        final_loss = recon_kfold_loss / folds
        nprint("Final loss {} [work to minimize this with the best settings ..]".format(final_loss) )
    
        self.ae_model = autoencoder
#
    def visualize_dimred_results(self, mode='pca') :
        if(mode == 'pca') :
            cols_ = ['PC0','PC1','PC2','PC3','PC4','PC5']
        else :
            cols_ = ['AE0','AE1','AE2','AE3','AE4','AE5']

        default_colors = self.test_df['default'].apply(lambda x : 'red' if x == 1 else 'green')

        scatter_matrix(self.test_df[cols_], alpha=0.1, figsize=[10,10], grid=True, c=default_colors, diagonal='kde')
        #scatter_matrix(encode_X[cols_], alpha=0.4, figsize=[10,10], grid=True)

        #marker='o',c=pcomps.Churn.apply(lambda x:churn_colors[x]
        plt.show()


    def drop_pca_ae_cols(self, df) :
        drop_cols = [x for x in df.columns if 'PC' in x]
        if(len(drop_cols) > 0) :
            df = df.drop(drop_cols,axis=1)
        drop_cols = [x for x in df.columns if 'AE' in x]
        if(len(drop_cols) > 0) :
            df = df.drop(drop_cols,inplace=True,axis=1)
        return df

    def update_train_test_df(self) :
        self.train_df = self.update_df(self.train_df, mode='train')
        self.test_df  = self.update_df(self.test_df, mode='test')

    def update_df(self, df, mode='train') :
        nprint("Starting update for {} dataframe ".format(mode))

        df = self.drop_pca_ae_cols(df)
        if(mode == 'train') :
            X_scaled = self.X_train_scaled
        else :
            X_scaled = self.X_test_scaled


        # Create Test data encoded values
        nprint("Adding PCA columns first")

        pca_encode_X = self.pca_model.transform(X_scaled)  #get the actual principal components as vectors
        pca_encode_X = pd.DataFrame(data=pca_encode_X)
        cols_ = {i:"PC"+str(i) for i in pca_encode_X.columns}
        nprint("Creating new columns : {}".format(cols_))
        pca_encode_X.rename(columns=cols_, inplace=True)

        nprint("Adding AE columns next")
        #outputs = [layer.output for layer in self.ae_model.layers]
        self.ae_model.summary()

        nprint("Grabbing AE Bottleneck layer")
        nl = len(self.ae_model.layers)
        nprint("Num Layers : {}".format(nl))

        nl = int((nl - 2)/2)  # strip off front/back ...find middle.

        btl_layer_str = 'dense_' + str(nl)
        nprint("Bottleneck Layer : {}".format(btl_layer_str))

        ae_bottleneck_model = Model(inputs=self.ae_model.input, outputs=self.ae_model.get_layer(btl_layer_str).output)
        ae_bottleneck_model.summary()

        ae_encode = ae_bottleneck_model.predict(x=X_scaled)
        ae_encode_X = pd.DataFrame(data=ae_encode, index=df.index)
        #print(ae_encode_X.head(5))
        cols_ = {i:"AE"+str(i) for i in ae_encode_X.columns}
        ae_encode_X.rename(columns=cols_, inplace=True)

        nprint(ae_encode_X.columns)

        nprint("Updating {} Dataframe ".format(mode))
        df = pd.concat([df.reset_index(),pca_encode_X,ae_encode_X.reset_index()],axis=1)

        return df


    def build_evaluate_dl_classifier(self, x_cols, regularization=0.001, epochs=3, batch_size = 16) :


        X = self.train_df[x_cols]
        Y = self.Y_train
        X_test = self.test_df[x_cols]
        Y_test = self.Y_test

        #Compare DL vs Logistic ....
        r = regularization
        
        input_layer = Input(shape=(X.shape[1], ), name="input_layer")
        fc0 = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(r), name="FC0")(input_layer)      
        #fc1 = Dense(5, activation= 'sigmoid', kernel_regularizer=regularizers.l2(r), name="FC1")(layer[-1]) 
        output_layer = Dense(1, activation= 'sigmoid', kernel_regularizer=regularizers.l2(r), name="final_layer")(fc0) 
        
                
        # construct and compile AE model
        dl_classifier = Model(input_layer, output_layer)
        dl_classifier.compile(optimizer='sgd', loss='mean_squared_error')
          
        
        dl_classifier.summary()
        fit_err = dl_classifier.fit(X, Y, shuffle=False, epochs=epochs, batch_size = batch_size,verbose=True, validation_data=(X_test,Y_test)) 

        Y_test_predict = np.where(dl_classifier.predict(x=X_test) > 0.5, 1, 0 )

        cnf_matrix =confusion_matrix(Y_test, Y_test_predict)
        class_names =  ['Default','Paid']  
        plot_confusion_matrix(cnf_matrix, class_names)

