import pandas as pd
from sklearn.cluster import DBSCAN
from itertools import compress
def pdutil_describeItemRange(df):  # Show the number of unique values in each column
    """pdutil_describeItemRange performs is similar to descrive for a DataFrame"""
    """but it displays the number of unique values in each column"""
    """argument is a dataFrame """
    """return is smaller dataFrame with the same columns as the original"""
    cols = df.columns.tolist()
    vals = pd.DataFrame ( [ len(set(df[s])) for s in df.columns.tolist()] ).T
    vals.columns = cols
    return vals

def pdutil_factorize(df):
    """pdutil_factorize performs converts every categorical column in a dataFrame to numeric values"""
    """pass in a DataFrame"""
    """return value is a similar dataFrame but with categorical values converted to numbers"""    
    return  df.apply(lambda x: pd.factorize(x)[0])

def pdutil_OneHotEncode(pdold, StaticCols, EncodeCols):
    """pdutil_OneHotEncode performs OneHotEncoding on specified columns in our dataframe"""
    """pdold is the old dataframe we are paasing in"""
    """StaticCols is a list of columns we do not wish to encode"""
    """EncodeCols is a list of columns we wish to encode"""
    """Returns a modified DataFrame with oneHotEncoded columns"""
    # make sure that we our static list is not in our encode list
    cols_OneHotEncode = sorted(list(set(EncodeCols) - set(StaticCols)))
    # create a new dataframe which represents the static columns
    pdnew = pdold[StaticCols]
    # loop thru the endcode list and OneHotEncode (using get_dummies) each column in the encode list
    # note - this will typically map a single column to several columns - one for each value contained in the original column
    for col in cols_OneHotEncode:
        new_names = []
        enc = pd.get_dummies(pdold[col])
        for subcol in  enc:
            name = col + str(subcol)
            new_names.append(name)
        enc.columns = new_names
        # concat the new OneHotEnoded dataframe to the original
        pdnew = pd.concat([pdnew, enc], axis=1)
    return pdnew


def pdutil_dbscan(df, eps_):
# Compute DBSCAN
    labels = []
    db = DBSCAN(eps=eps_, min_samples=100).fit(df)
    labels = pd.DataFrame(db.labels_ )
    #n_clusters_ = labels[0].madf()
    n_clusters_ = len(list(set(db.labels_))) - (1 if -1 in db.labels_ else 0)
    # Number of clusters in labels, ignoring noise if present.
    tdft = 'dbsPCA'
    df[tdft]=labels
    print('eps:{} Estimated number of clusters: {}'.format(eps_, n_clusters_) ) 
    return df, labels, n_clusters_
