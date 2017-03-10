import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# common global variables
# helpful in ensembling
with open('data/training_impro.tsv','r') as tsv:
    mat = [line.strip().split('\t') for line in tsv]
#mat = pd.read_csv('data/training_impro.tsv',sep='\t')
#print mat.ix[:,'y']
# ignoring first row of data x,y
i=1
X=[]
y=[]
while i<len(mat):
    if len(mat[i])!=2:
        mat[i] = [''] + mat[i]
    X.append(mat[i][0])
    y.append(1 if mat[i][1] == 'Found' else 0)
    i=i+1
#dividing training and testing
split = 7000
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

with open('data/eval_data.txt', 'r') as f:
    x = f.readlines()
i=0
while i<len(x):
    x[i]=x[i][:-1] # ---- removing '\n' ----
    i=i+1

# ------------- Fitting on all dataset -----------------------------------------
# ------------- Hence taking both training_data.tsv & eval_text ----------------
X_all = X + x
# ------ This is done so matrix size matches while .(dot) multiplication -------
#print len(X), len(X_train), len(X_test)
#print len(y), len(y_train), len(y_test)

count_vect = CountVectorizer()
count_vect.fit(X_all)
X_all_counts = count_vect.transform(X_all)
X_train_counts = count_vect.transform(X_train)

# ----- term frequency (TF) inverse doc frequency ------------------

tfidf_transformer = TfidfTransformer().fit(X_all_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
# ------------------------------------------------------------------------------
def naive():
    # ---------------training naive bayes classifier ---------------------------

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # ---------------predict on hold out dataset -------------------------------
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print X_train_tfidf.shape, X_test_tfidf.shape, X_all_counts.shape
    predicted = clf.predict(X_test_tfidf)
    joblib.dump(clf, 'models/nb.pkl')
    print len(predicted)
    #for doc, category in zip(X_test, predicted):
    #     print('%r => %s' % (doc, 'Found' if category == 1 else 'Not Found'))

    # -------------checking quality of classifier------------------------------

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    j = 0
    while j < len(X_test):
        if predicted[j] == 0 and y_test[j] == 0:
            tn = tn + 1
        if predicted[j] == 0 and y_test[j] == 1:
            fn = fn + 1
        if predicted[j] == 1 and y_test[j] == 1:
            tp = tp + 1
        if predicted[j] == 1 and y_test[j] == 0:
            fp = fp + 1
        j = j + 1
    print 'tp ', tp
    print 'tn ', tn
    print 'fp ', fp
    print 'fn ', fn
    precision = (float(tp)) / (float(tp) + float(fp))
    recall = (float(tp)) / (float(tp) + float(fn))
    F1 = 2*precision*recall/(precision+recall)
    accuracy = float(tp+tn) / float(tp+tn+fp+fn)
    print 'accuracy ', accuracy
    print 'F1 ', F1
    print 'precision', precision
    print 'recall ', recall
    print '----------------'

def svm():
    # ---------------training Support Vector Machine classifier ----------------

    clf = SVC(kernel="linear").fit(X_train_tfidf, y_train)

    # ---------------predict on hold out dataset -------------------------------
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print X_train_tfidf.shape, X_test_tfidf.shape, X_all_counts.shape
    predicted = clf.predict(X_test_tfidf)
    joblib.dump(clf, 'models/svm.pkl')
    print len(predicted)
    #for doc, category in zip(X_test, predicted):
    #     print('%r => %s' % (doc, 'Found' if category == 1 else 'Not Found'))

    # -------------checking quality of classifier------------------------------

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    j = 0
    while j < len(X_test):
        if predicted[j] == 0 and y_test[j] == 0:
            tn = tn + 1
        if predicted[j] == 0 and y_test[j] == 1:
            fn = fn + 1
        if predicted[j] == 1 and y_test[j] == 1:
            tp = tp + 1
        if predicted[j] == 1 and y_test[j] == 0:
            fp = fp + 1
        j = j + 1
    print 'tp ', tp
    print 'tn ', tn
    print 'fp ', fp
    print 'fn ', fn
    precision = (float(tp)) / (float(tp) + float(fp))
    recall = (float(tp)) / (float(tp) + float(fn))
    F1 = 2*precision*recall/(precision+recall)
    accuracy = float(tp+tn) / float(tp+tn+fp+fn)
    print 'accuracy ', accuracy
    print 'F1 ', F1
    print 'precision', precision
    print 'recall ', recall
    print '----------------'

def mlp():
    # ---------------training Neural Network MLP classifier --------------------

    clf =  MLPClassifier(hidden_layer_sizes=40, alpha=1, max_iter=50, solver='adam').fit(X_train_tfidf, y_train)

    # ---------------predict on hold out dataset -------------------------------
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print X_train_tfidf.shape, X_test_tfidf.shape, X_all_counts.shape
    predicted = clf.predict(X_test_tfidf)
    joblib.dump(clf, 'models/mlp.pkl')
    print len(predicted)
    #for doc, category in zip(X_test, predicted):
    #     print('%r => %s' % (doc, 'Found' if category == 1 else 'Not Found'))

    # -------------checking quality of classifier------------------------------

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    j = 0
    while j < len(X_test):
        if predicted[j] == 0 and y_test[j] == 0:
            tn = tn + 1
        if predicted[j] == 0 and y_test[j] == 1:
            fn = fn + 1
        if predicted[j] == 1 and y_test[j] == 1:
            tp = tp + 1
        if predicted[j] == 1 and y_test[j] == 0:
            fp = fp + 1
        j = j + 1
    print 'tp ', tp
    print 'tn ', tn
    print 'fp ', fp
    print 'fn ', fn
    precision = (float(tp)) / (float(tp) + float(fp))
    recall = (float(tp)) / (float(tp) + float(fn))
    F1 = 2*precision*recall/(precision+recall)
    accuracy = float(tp+tn) / float(tp+tn+fp+fn)
    print 'accuracy ', accuracy
    print 'F1 ', F1
    print 'precision', precision
    print 'recall ', recall
    print '----------------'

def ada():
    # ---------------training adaboost classifier --------------------

    clf =  AdaBoostClassifier().fit(X_train_tfidf, y_train)

    # ---------------predict on hold out dataset -------------------------------
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print X_train_tfidf.shape, X_test_tfidf.shape, X_all_counts.shape
    predicted = clf.predict(X_test_tfidf)
    joblib.dump(clf, 'models/adaboost.pkl')
    print len(predicted)
    #for doc, category in zip(X_test, predicted):
    #     print('%r => %s' % (doc, 'Found' if category == 1 else 'Not Found'))

    # -------------checking quality of classifier------------------------------

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    j = 0
    while j < len(X_test):
        if predicted[j] == 0 and y_test[j] == 0:
            tn = tn + 1
        if predicted[j] == 0 and y_test[j] == 1:
            fn = fn + 1
        if predicted[j] == 1 and y_test[j] == 1:
            tp = tp + 1
        if predicted[j] == 1 and y_test[j] == 0:
            fp = fp + 1
        j = j + 1
    print 'tp ', tp
    print 'tn ', tn
    print 'fp ', fp
    print 'fn ', fn
    precision = (float(tp)) / (float(tp) + float(fp))
    recall = (float(tp)) / (float(tp) + float(fn))
    F1 = 2*precision*recall/(precision+recall)
    accuracy = float(tp+tn) / float(tp+tn+fp+fn)
    print 'accuracy ', accuracy
    print 'F1 ', F1
    print 'precision', precision
    print 'recall ', recall
    print '----------------'

def rfc():
    # ---------------training Random Forest classifier --------------------

    clf =  RandomForestClassifier(max_depth=15, n_estimators=50, max_features=1).fit(X_train_tfidf, y_train)

    # ---------------predict on hold out dataset -------------------------------
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print X_train_tfidf.shape, X_test_tfidf.shape, X_all_counts.shape
    predicted = clf.predict(X_test_tfidf)
    joblib.dump(clf, 'models/rfc.pkl')
    print len(predicted)
    #for doc, category in zip(X_test, predicted):
    #     print('%r => %s' % (doc, 'Found' if category == 1 else 'Not Found'))

    # -------------checking quality of classifier------------------------------

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    j = 0
    while j < len(X_test):
        if predicted[j] == 0 and y_test[j] == 0:
            tn = tn + 1
        if predicted[j] == 0 and y_test[j] == 1:
            fn = fn + 1
        if predicted[j] == 1 and y_test[j] == 1:
            tp = tp + 1
        if predicted[j] == 1 and y_test[j] == 0:
            fp = fp + 1
        j = j + 1
    print 'tp ', tp
    print 'tn ', tn
    print 'fp ', fp
    print 'fn ', fn
    precision = (float(tp)) / (float(tp) + float(fp))
    recall = (float(tp)) / (float(tp) + float(fn))
    F1 = 2*precision*recall/(precision+recall)
    accuracy = float(tp+tn) / float(tp+tn+fp+fn)
    print 'accuracy ', accuracy
    print 'F1 ', F1
    print 'precision', precision
    print 'recall ', recall
    print '----------------'
#naive()
#svm()
#mlp()
#ada()
rfc()
