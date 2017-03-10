from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.parse.stanford import StanfordDependencyParser
path_to_jar = '../../LBS/LBS-X/lib/stanford-parser/stanford-parser.jar'
path_to_models_jar = '../../LBS/LBS-X/lib/stanford-parser/stanford-parser-3.6.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
from nltk.parse.stanford import StanfordParser
parser=StanfordParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

from nltk.tree import ParentedTree

# common global variables
# helpful in ensembling

with open('data/training_impro.tsv','r') as tsv:
    mat = [line.strip().split('\t') for line in tsv]
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

with open('data/eval_data.txt', 'r') as f:
    x = f.readlines()
i=0
while i<len(x):
    x[i]=x[i][:-1] # ---- removing '\n' ----
    i=i+1


# ------------- Fitting on all dataset -----------------------------------------
# ------------- Hence taking both training_data.tsv & eval_text ----------------
X_all = X + x
count_vect = CountVectorizer()
count_vect.fit(X_all)
X_all_counts = count_vect.transform(X_all)
tfidf_transformer = TfidfTransformer().fit(X_all_counts)

def classify():
    clf = joblib.load('models/nb.pkl')
    with open('data/eval_data.txt', 'r') as f:
        x = f.readlines()
    i=0
    while i<len(x):
        x[i]=unicode(x[i][:-1],'utf-8') # ---- removing '\n' ----
        i=i+1
    print len(x), type(x)

    # ------------- transform TFIDF --------------------------------------------

    X_test_counts = count_vect.transform(x)
    print X_test_counts.shape
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    print X_test_tfidf.shape

    predicted = clf.predict(X_test_tfidf)
    print len(x), len(predicted)
    #print predicted

def remove_empty(res, resf):
    i=0
    while i < len(res):
        if res[i] == []:
            del res[i]
            del resf[i]
        i=i+1
    return res, resf

def bruteforce(each_ex):
    return each_ex

def traverse(t, all_leaves, response, response_ref):
    try:
        t.label()
    except AttributeError:
        return
    else:
        for child in t:
            #print "height ", t.height()
            #print "label ", t.label()
            #print "child ", child
            #print "parent ", t.parent()
            #if t.parent() != None:
            #    print "leaver ", t.parent().leaves()
            #print type(child), type(t.label), type(t.parent)
            try:
                if child.label() in ['NP']:
                    temp = t
                    while temp.label() in ['S', 'VB', 'PP']:
                        temp = t.parent()
                    if temp.label() in ['VP', 'NP']:
                        #print '~~~~~~~~ inside .... ~~~~~~~ '
                        if temp.parent().leaves() != all_leaves:
                            #print '~~~~~~~~ inside 2 .... ~~~~~~~ '
                            asib = temp.parent().leaves()
                            rsib=None
                            try:
                                rsib = child.right_sibling().leaves()
                            except Exception:
                                fuck='it'
                            #print '~~~~ ', asib, rsib
                            if rsib != None:
                                #print '~~~~~~~~ inside 3 .... ~~~~~~~ '
                                if temp.parent().label() != None:
                                    response.append(asib[1:asib.index(rsib[0])])
                                else:
                                    response.append(asib[:asib.index(rsib[0])])
                            else:
                                #print '~~~~~~~~ inside 4 .... ~~~~~~~ '
                                response.append(asib)
                            flag = str(child).split()
                            fflag = ' '.join(flag)
                            response_ref.append(fflag)

            except Exception:
                fuck='it'
            traverse(child, all_leaves, response, response_ref)
    return response, response_ref

def reminder_phrase(each_ex):
    # --- input: <String> each_ex
    # --- return: <String> rem_ex
    #result = dependency_parser.raw_parse(each_ex)
    #dep = result.next()
    rem_ex = ''
    a = list(parser.raw_parse(each_ex))
    # NLTK tree
    pt = ParentedTree.convert(a[0])
    print type(pt), len(pt), pt
    print 'tree end '
    response = []
    response_ref = []
    res, resf = traverse(pt, pt.leaves(), response, response_ref)
    res, resf = remove_empty(res, resf)
    vote=[]
    if len(res) == 1:
        rem_ex = ' '.join(res[0])
        if rem_ex == '':
            rem_ex = bruteforce(each_ex)
    elif len(res) > 1:
        i=0
        for each in resf:
            vote.append(sum(each.count(x) for x in ['NN', 'NNS', 'JJ']))
            neg = sum(each.count(x) for x in ['CD', 'DT', 'IN'])
            vote[i] = vote[i] - neg
            i=i+1
        rem_ex = ' '.join(res[vote.index(max(vote))])
    else:
        rem_ex = bruteforce(each_ex)
    print 'res ', res
    print 'ref ', resf
    print vote
    print '* ', rem_ex
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

reminder_phrase('Set a reminder on 4 th Dec of going to meet sonal miss at 2:00 pm')
reminder_phrase('Remind me to purchase shoe polish liquid Date:3 Jan Time:6.30 pm')
reminder_phrase('Please remind me for internal audit review meeting at 12.45 today')
reminder_phrase('And a reminder tomorrow at 11.30 am to go through basic codings and share markets.')
reminder_phrase('Please remind me on Tuesday that I have an appointment at YLG for hair spa at 4.15')
reminder_phrase('Thanks at least I\'ll remember my loves birthday this time')
reminder_phrase('Susan dmello meeting with sujit sir remind him Tomorrow')
reminder_phrase('I need to set reminder to msg babbu at 7 in evening')
reminder_phrase('Hi give me a reminder to pay LIC Premium on tonight 9 PM')
reminder_phrase('Remind me to go to bank at 11 am today')
reminder_phrase('Remind me to buy eggs on next Monday and Tuesday at 9pm')

#reminder_phrase('')


#classify()
