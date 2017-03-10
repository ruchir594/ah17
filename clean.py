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
