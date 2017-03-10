def clean():
    with open('data/eval_data.txt', 'r') as f:
        x = f.readlines()
    with open('raw_eval.txt', 'r') as f:
        y = f.readlines()
    with open('eval_predict.tsv', 'w') as f:
        i=0
        print len(x), len(y)
        count = 0
        while i<len(x):
            #f.write(x[i][:-1] + '\t' + y[i][:-1] + '\n')
            if y[i][:-1] != "Not Found":
                count=count+1
            i=i+1
    print count


clean()
