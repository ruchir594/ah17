# ------------------------------------------------------------------------------
# raw_pred_ens = output of Classifier
# raw_eval = output of Parser

# the reason for doing separate parser output is that it helps a LOT,
# It makes testing a classifier much much faster.

# raw_eval file already contains possible reminder text for each message
# This function "clean" reads the input of raw_pred_ens, which is a binary
# vector of size of eval_data, if the value of raw_pred_ens is 1, reminder text
# from that message much be found.

# output
# gives eval_pred_ens file which is in TSV format. 


# ------------------------------------------------------------------------------
def clean():
    with open('data/eval_data.txt', 'r') as f:
        x = f.readlines()
    with open('raw_eval.txt', 'r') as f:
        y = f.readlines()
    with open('raw_pred_ens.txt', 'r') as f:
        z = f.readlines()
    with open('eval_predict_ens.tsv', 'w') as f:
        i=0
        print len(x), len(y), len(z)
        count = 0
        while i<len(x):
            if int(z[i]) == 1:
                f.write(x[i][:-1] + '\t' + y[i][:-1] + '\n')
            else:
                f.write(x[i][:-1] + '\tNot Found\n')
            if y[i][:-1] != "Not Found":
                count=count+1
            i=i+1
    print count


clean()
