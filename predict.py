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
import re, nltk
import numpy as np
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def getWords(data):
    return re.compile(r"[\w']+").findall(data)

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
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
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
    print np.count_nonzero(predicted)
    with open('raw_pred.txt','w') as f:
        for each in predicted:
            f.write(str(each)+'\n')
    #print predicted
    """reminder_text = []
    i=0
    with open('raw_eval.txt', 'w') as rev:
        for each in predicted:
            if each == 0:
                a_temp = 'Not Found'
                a_temp = str(a_temp)
            else:
                a_temp = str(reminder_phrase(x[i]))
                print a_temp, type(a_temp)
            reminder_text.append(a_temp)
            rev.write(a_temp+'\n')
            print i
            i=i+1
    with open('eval_predict.tsv', 'w') as f:
        i=0
        while i < len(x):
            f.write(x[i] + '\t' + reminder_phrase[i] + '\n' )
            i=i+1"""

def pred_all_eval():
    with open('data/eval_data.txt', 'r') as f:
        x = f.readlines()
    i=0
    with open('raw_eval.txt', 'w') as rev:
        for each in x:
            rev.write(str(reminder_phrase(re.sub(r'[^\x00-\x7f]',r'', x[i])))+'\n')
            print i
            i=i+1
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def remove_empty(res, resf):
    i=0
    while i < len(res):
        if res[i] == []:
            del res[i]
            del resf[i]
        i=i+1
    return res, resf

# ------------------------------------------------------------------------------
# ------- recursive function to extract POS tagg from  Parse tree --------------
# ------------------------------------------------------------------------------
def get_pos(t, words, pos):
    for subtree in t:
        if type(subtree) == nltk.tree.Tree:
            if subtree.height() == 2:   #child nodes
                pos.append(str(subtree).replace("(", "").replace(")", "").split())
            get_pos(subtree, words, pos)
    return pos

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def bruteforce(each_ex, t):
    re_ex=''
    re_words=['reminder', 'remind']
    pos = get_pos(t, getWords(each_ex), [])
    hold = 3
    for each in pos:
        if each[0] in ['IN', 'PRP', 'TO', 'DT', 'CD']:
            del pos[pos.index(each)]
        if each[1].lower() in re_words:
            hold = pos.index(each)
    i=0
    while i<len(pos):
        if i>hold-3 and i<hold+3:
            re_ex = re_ex + pos[i][1] + ' '
        i=i+1
    if re_ex == '':
        re_ex = 'Not Found'
    return re_ex
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
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
                    j=0
                    while temp.label() in ['S', 'VB', 'PP']:
                        temp = t.parent()
                        j=j+1
                        if j>5:
                            break
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
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def reminder_phrase(each_ex):
    # --- input: <String> each_ex
    # --- return: <String> rem_ex
    #result = dependency_parser.raw_parse(each_ex)
    #dep = result.next()
    rem_ex = ''
    a = list(parser.raw_parse(each_ex))
    # NLTK tree
    pt = ParentedTree.convert(a[0])
    #print type(pt), len(pt), pt
    #print 'tree end '
    response = []
    response_ref = []
    res, resf = traverse(pt, pt.leaves(), response, response_ref)
    res, resf = remove_empty(res, resf)
    vote=[]
    if len(res) == 1:
        rem_ex = ' '.join(res[0])
        if rem_ex == '':
            rem_ex = bruteforce(each_ex, a[0])
    elif len(res) > 1:
        i=0
        for each in resf:
            vote.append(sum(each.count(x) for x in ['NN', 'NNS', 'JJ']))
            neg = sum(each.count(x) for x in ['CD', 'DT', 'IN'])
            vote[i] = vote[i] - neg
            i=i+1
        rem_ex = ' '.join(res[vote.index(max(vote))])
    else:
        rem_ex = bruteforce(each_ex, a[0])
    #print 'res ', res
    #print 'ref ', resf
    #print vote
    #print '* ', rem_ex
    return rem_ex
    #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

"""print reminder_phrase('Set a reminder on 4 th Dec of going to meet sonal miss at 2:00 pm')
print reminder_phrase('Remind me to purchase shoe polish liquid Date:3 Jan Time:6.30 pm')
print reminder_phrase('Please remind me for internal audit review meeting at 12.45 today')
print reminder_phrase('And a reminder tomorrow at 11.30 am to go through basic codings and share markets.')
print reminder_phrase('Please remind me on Tuesday that I have an appointment at YLG for hair spa at 4.15')
print reminder_phrase('Thanks at least I\'ll remember my loves birthday this time')
print reminder_phrase('Susan dmello meeting with sujit sir remind him Tomorrow')
print reminder_phrase('I need to set reminder to msg babbu at 7 in evening')
print reminder_phrase('Hi give me a reminder to pay LIC Premium on tonight 9 PM')
print reminder_phrase('Remind me to go to bank at 11 am today')
print reminder_phrase('Remind me to buy eggs on next Monday and Tuesday at 9pm')"""

#print reminder_phrase('')


classify()
pred_all_eval()
