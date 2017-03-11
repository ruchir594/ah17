import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.externals import joblib
from collections import Counter

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
    # writing in file for predict.py to use
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
    # writing in file for predict.py to use
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
    # writing in file for predict.py to use
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
    # writing in file for predict.py to use
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

def svm_nl():
    # ---------------training RBF SVM classifier --------------------

    clf =  SVC(gamma=2, C=1).fit(X_train_tfidf, y_train)

    # ---------------predict on hold out dataset -------------------------------
    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print X_train_tfidf.shape, X_test_tfidf.shape, X_all_counts.shape
    predicted = clf.predict(X_test_tfidf)
    # writing in file for predict.py to use
    joblib.dump(clf, 'models/svm_nl.pkl')
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


# ------------------------------------------------------------------------------
# The following function is used to measure the effectiveness of ensembling
# We make multiple combination of models vote on final outcome of classifier
# Measure accuracy on X_Test
# ------------------------------------------------------------------------------

def ens_x_text():
    clf1 = joblib.load('models/nb.pkl')
    clf2 = joblib.load('models/svm.pkl')
    clf3 = joblib.load('models/svm_nl.pkl')
    clf4 = joblib.load('models/nn.pkl')
    clf5 = joblib.load('models/adaboost.pkl')
    with open('data/eval_data.txt', 'r') as f:
        x = f.readlines()
    i=0
    while i<len(x):
        x[i]=unicode(x[i][:-1],'utf-8') # ---- removing '\n' ----
        i=i+1
    print len(x), type(x)

    # ------------- transform TFIDF --------------------------------------------

    X_test_counts = count_vect.transform(X_test)
    print X_test_counts.shape
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    print X_test_tfidf.shape

    predicted1 = clf1.predict(X_test_tfidf)
    predicted2 = clf2.predict(X_test_tfidf)
    predicted3 = clf3.predict(X_test_tfidf)
    predicted4 = clf4.predict(X_test_tfidf)
    predicted5 = clf5.predict(X_test_tfidf)
    print len(x), len(predicted1)
    predicted=[]
    with open('raw_pred_ens.txt','w') as f:
        # ~~ as Section 1.6.U in ./docs/approach, a voting of 5 best algo takes place
        for w1,w2,w3,w4,w5 in zip(predicted1, predicted2, predicted3, predicted4, predicted5):
            c = Counter([w2,w3,w4,w5]) notice, counting vote of only 4 !!!!!!!!
            value, count = c.most_common()[0]
            predicted.append(value)
    print len(predicted)
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

naive()
svm()
mlp()
ada()
svm_nl()
#ens_x_text()
