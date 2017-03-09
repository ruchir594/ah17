import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
def naive():
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
        X.append(unicode(mat[i][0],'utf-8'))
        y.append(1 if mat[i][1] == 'Found' else 0)
        i=i+1
    #dividing training and testing
    X_train = X[:7000]
    y_train = y[:7000]
    X_test = X[7000:]
    y_test = y[7000:]
    #print len(X), len(X_train), len(X_test)
    #print len(y), len(y_train), len(y_test)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    #print X_train_counts.shape

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)


    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predicted = clf.predict(X_test_tfidf)
    print len(predicted)
    #for doc, category in zip(X_test, predicted):
    #     print('%r => %s' % (doc, 'Found' if category == 1 else 'Not Found'))
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
