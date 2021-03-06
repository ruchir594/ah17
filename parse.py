from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.parse.stanford import StanfordDependencyParser
# ------------------------------------------------------------------------------
# CHANGE THIS VALUE ACCORDING TO YOUR CONFIG
# ------------------------------------------------------------------------------
path_to_jar = '../../LBS/LBS-X/lib/stanford-parser/stanford-parser.jar'
path_to_models_jar = '../../LBS/LBS-X/lib/stanford-parser/stanford-parser-3.6.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
from nltk.parse.stanford import StanfordParser
parser=StanfordParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
from collections import Counter

from nltk.tree import ParentedTree
import re, nltk
import numpy as np
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def getWords(data):
    return re.compile(r"[\w']+").findall(data)

# ------------------------------------------------------------------------------
# reads every message in eval_data and writes a potential reminder text
# this is done is make testing of a classifier faster
# for final output, if the output of ensemple classifier is 0, Not Found value
# is associated to that message text. if 1, we just take the value from
# raw_eval as it makes testing a Classifier much much faster.
# ------------------------------------------------------------------------------
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
# removed an empty element in array res
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
# this funcation is called as a last resort.
# when our Parser based approach cannot find the response text, we turn to
# POS Tagger approach and a bit of hacking.
# ------------------------------------------------------------------------------
def bruteforce(each_ex, t):
    re_ex=''
    re_words=['reminder', 'remind', 'reminded']
    pos = get_pos(t, getWords(each_ex), [])
    # initialized to average median
    hold = 3
    for each in pos:
        # we want to remove words with inderior tags that are likely to not
        # carry meaningful information
        if each[0] in ['IN', 'PRP', 'TO', 'DT', 'CD']:
            del pos[pos.index(each)]
        # Try to find the activation
        # this can potentially be improved by using Word2Vec
        if each[1].lower() in re_words:
            hold = pos.index(each)
    i=0
    # find 4 words before and after the activated word
    while i<len(pos):
        if i>hold-3 and i<hold+3:
            re_ex = re_ex + pos[i][1] + ' '
        i=i+1
    if re_ex == '':
        re_ex = 'Not Found'
    return re_ex
# ------------------------------------------------------------------------------
# Input:
# t: Parse Tree
# all_leaver: a List with all the leaves in the tree
# response: Empty Array
# response_ref: Empty Array

# Output:
# response: An array with possible leaves of a subtree
# response_ref: Leaves with POS information to each leaf

# Based on Section 2.2 in ./doc/approach
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
                # ~~ Section 2.2.1 in ./doc/approach
                if child.label() in ['NP']:
                    temp = t
                    j=0
                    # ~~ Section 2.2.2 in ./doc/approach
                    while temp.label() in ['S', 'VB', 'PP']:
                        temp = t.parent()
                        j=j+1
                        if j>5:
                            break
                    # ~~ Section 2.2.3 in ./doc/approach
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
                            # ~~ Section 2.2.4 & 2.2.5 in ./doc/approach
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
            # recursive call
            traverse(child, all_leaves, response, response_ref)
    return response, response_ref
# ------------------------------------------------------------------------------
# input
# Each_ex: Message text input

# output
# rem_ex: reminder text
# ------------------------------------------------------------------------------
def reminder_phrase(each_ex):
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
            # ~~ Section 2.4 Voting for best chunk
            vote.append(sum(each.count(x) for x in ['NN', 'NNS', 'JJ']))
            neg = sum(each.count(x) for x in ['CD', 'DT', 'IN'])
            # positive votes are appended. And negative votes are subtracted
            # from same var place
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

pred_all_eval()
