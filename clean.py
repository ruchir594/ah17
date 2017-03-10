def clean():
    with open('data/eval_data.txt', 'r') as f:
        x = f.readlines()
    with open('raw_eval.txt', 'r') as f:
        y = f.readlines()
    with open('eval_predict.tsv', 'w') as f:
        i=0
        print len(x), len(y)
        while i<len(x):
            f.write(x[i][:-1] + '\t' + y[i][:-1] + '\n')
            i=i+1

clean()
