from itertools import compress
class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(compress(self, index))
        
def DescribeItemRange(df):  # Show the number of unique values in each column
    cols = df.columns.tolist()
    vals = pd.DataFrame ( [ len(set(df[s])) for s in df.columns.tolist()] ).T
    vals.columns = cols
    return vals

def ReplaceNans (df):  #replace NaNs in all df locations with zeros
    df.fillna(0)
    return df

def SelectZeroColumns (df): #return a list with column names of zero filled columns
    dropcols = (df == 0).all().astype(int).tolist()
    return [s for s in mylist[dropcols]  ]

def SelectNonZeroColumns(df):  #return a list with columns names of nonzero columns
    return list(set(StringifyColumnNames(df))- set(SelectZeroColumns(df) ))

def StringifyColumnNames (df):   # handle unicode strings and covert to normal strings
    cols = df.columns.tolist()
    cols = [str(r) for r in cols]
    return cols

def Factorize(df):
    return  df.apply(lambda x: pd.factorize(x)[0])

def MyOneHotEncode(pdold, StaticCols, EncodeCols):
    """MyOneHotEncode performs OneHotEncoding on specified columns in our dataframe"""
    """pdold is the old dataframe we are paasing in"""
    """StaticCols is a list of columns we do not wish to encode"""
    """EncodeCols is a list of columns we wish to encode"""
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


def my_dbscan(df, eps_):
# Compute DBSCAN
    labels = []
    db = DBSCAN(eps=eps_, min_samples=100).fit(df)
    labels = pd.DataFrame(db.labels_ )
    #n_clusters_ = labels[0].madf()
    n_clusters_ = len(list(set(db.labels_))) - (1 if -1 in db.labels_ else 0)
    # Number of clusters in labels, ignoring noise if present.
    #tdft = 'dbsPCA_' + str(n_clusters_)
    tdft = 'dbsPCA'
    df[tdft]=labels
    print('eps:{} Estimated number of clusters: {}'.format(eps_, n_clusters_) ) 
    return df, labels, n_clusters_
