with open('data/training_data.tsv','r') as tsv:
    mat = [line.strip().split('\t') for line in tsv]
    x=[]
    y=[]
    for each in mat:
        if len(each)==1:
            x.append('')
            if each[0] == 'Not Found':
                y.append('Not Found')
            else:
                y.append('Found')
        else:
            if each[1] == 'Not Found':
                y.append('Not Found')
            else:
                y.append('Found')
            x.append(each[0])
    print len(x), len(y)
with open('data/training_impro.tsv', 'w') as f:
    i=0
    f.write('x\ty\n')
    while i<len(x):
        f.write(str(x[i]) + '\t' + str(y[i]) + '\n')
        i=i+1
