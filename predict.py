from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

def get_postagging(parsedData):
    full_pos = []
    sent = []
    for span in parsedData.sents:
        sent = sent + [parsedData[i] for i in range(span.start, span.end)]
        #break

    for token in sent:
        full_pos.append([token.orth_, token.pos_])
    return full_pos

def get_dependency(parsedEx):
    # Let's look at the dependencies of this example:
    # shown as: original token, dependency tag, head word, left dependents, right dependents
    full_dep = []
    for token in parsedEx:
        full_dep.append([token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights]])
    return full_dep

from stat_parser import Parser
parser = Parser()
def reminder_phrase(each_ex):
    # --- input: <String> each_ex
    # --- return: <String> rem_ex
    print parser.parse(each_ex)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

reminder_phrase('Set a reminder on 4 th Dec of going to meet sonal miss at 2:00 pm')
reminder_phrase('Remind me to purchase shoe polish liquid Date:3 Jan Time:6.30 pm')
reminder_phrase('Please remind me for internal audit review meeting at 12.45 today')
reminder_phrase('And a reminder tomorrow at 11.30 am to go through basic codings and share markets.')
reminder_phrase('')


classify()
