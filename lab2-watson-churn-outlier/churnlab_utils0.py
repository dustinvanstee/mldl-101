import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from itertools import compress
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import tensorflow as tf
	
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    import numpy as np
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
    plt.savefig('confusion_mat.png', bbox_inches='tight', format='png', dpi=300, pad_inches=0,transparent=True)
    plt.show()
    return
	
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

## import tensorflow as tf
import numpy as np
import pandas as pd
import math

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np

CLASS_SIZE = 2
#DATA_SIZE = 0

def load_csv(filename):
    file = pd.read_csv(filename, header=0)

    # get sample's metadata
    n_samples = int(file.columns[0])
    n_features = int(file.columns[1])

    # divide samples into explanation variables and target variable
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,), dtype=np.int)
    for i, row in enumerate(file.itertuples()):
        target[i] = np.asarray(row[-1], dtype=np.int)
        data[i] = np.asarray(row[1:n_features+1], dtype=np.float64)
    return (data, target)

# output train data 
def get_batch_data(x_train, y_train, size=None):
    if size is None:
        size = len(x_train)
    batch_xs = x_train
    batch_ys = []

    # convert to 1-of-N vector
    for i in range(len(y_train)):
        val = np.zeros((CLASS_SIZE), dtype=np.float64)
        val[y_train[i]] = 1.0
        batch_ys.append(val)
    batch_ys = np.asarray(batch_ys)
    return batch_xs[:size], batch_ys[:size]

# output test data
def get_test_data(x_test, y_test):
    batch_ys = []

    # convert to 1-of-N vector
    for i in range(len(y_test)):
        val = np.zeros((CLASS_SIZE), dtype=np.float64)
        val[y_test[i]] = 1.0
        batch_ys.append(val)
    return x_test, np.asarray(batch_ys)

# for parameter initialize
def get_stddev(in_dim, out_dim):
    return 1.3 / math.sqrt(float(in_dim) + float(out_dim))

# DNN Model Class
class Classifier:
    def __init__(self, hidden_units=[10], n_classes=0, data_size = 0):
        self._hidden_units = hidden_units
        self._n_classes = n_classes
        self._data_size = data_size
        self._sess = tf.Session()

    # build model
    def inference(self, x):
        hidden = []

        # Input Layer
        with tf.name_scope("input"):
            weights = tf.Variable(tf.truncated_normal([self._data_size , self._hidden_units[0]], stddev=get_stddev(self._data_size, self._hidden_units[0]), seed=42), name='weights')
            biases = tf.Variable(tf.zeros([self._hidden_units[0]]), name='biases')
            input = tf.matmul(x, weights) + biases

        # Hidden Layers
        for index, num_hidden in enumerate(self._hidden_units):
            if index == len(self._hidden_units) - 1: break
            with tf.name_scope("hidden{}".format(index+1)):
                weights = tf.Variable(tf.truncated_normal([num_hidden, self._hidden_units[index+1]], seed=42, stddev=get_stddev(num_hidden, self._hidden_units[index+1])), name='weights')
                biases = tf.Variable(tf.zeros([self._hidden_units[index+1]]), name='biases')
                inputs = input if index == 0 else hidden[index-1]
                hidden.append(tf.nn.relu(tf.matmul(inputs, weights) + biases, name="hidden{}".format(index+1)))
        
        # Output Layer
        with tf.name_scope('output'):
            weights = tf.Variable(tf.truncated_normal([self._hidden_units[-1], self._n_classes], seed=42, stddev=get_stddev(self._hidden_units[-1], self._n_classes)), name='weights')
            biases = tf.Variable(tf.zeros([self._n_classes]), name='biases')
            logits = tf.nn.softmax(tf.matmul(hidden[-1], weights) + biases)

        return logits

    # loss function
    def loss(self, logits, y):        
        #return -tf.reduce_mean(y * tf.log(logits))
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # fitting function for train data
    def fit(self, x_train=None, y_train=None, steps=200):
        # build model
        x = tf.placeholder(tf.float32, [None, self._data_size ])
        y = tf.placeholder(tf.float32, [None, CLASS_SIZE])
        logits = self.inference(x)
        loss = self.loss(logits, y)
        train_op = tf.train.AdamOptimizer(0.003).minimize(loss)

        # save variables
        self._x = x
        self._y = y
        self._logits = logits
 
        # init parameters
        #init = tf.initialize_all_variables() 
        init = tf.global_variables_initializer()
        self._sess.run(init)

        # train
        for i in range(steps):
            batch_xs, batch_ys = get_batch_data(x_train, y_train)
            self._sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

    # evaluation function for test data
    def evaluate(self, x_test=None, y_test=None):
        x_test, y_test = get_test_data(x_test, y_test)
        
        # build accuracy calculate step
        correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # evaluate
        return self._sess.run([accuracy], feed_dict={self._x: x_test, self._y: y_test})

    # label prediction
    def predict(self, samples):
        predictions = tf.argmax(self._logits, 1)
        return self._sess.run(predictions, {self._x: samples})
