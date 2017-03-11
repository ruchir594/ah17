# Find any reminder text from message text

### To understand the approach, head to [docs/approach.rtf](https://github.com/ruchir594/ah17/blob/master/docs/approach.rtf) in this repo.

## Adding files
### Make a directory "data" in this repo.
### Add following files...
  training_data.tsv

  eval_data.tsv

run
```
python convert.py
```
this will make a new TSV training_impro.tsv in ./data repository. This basically trims the training_data's second column to only "Found" or "Not Found"

Also run
```
pip install -r requirements.txt
```
Download Stanford [Dependency Parser](http://nlp.stanford.edu/software/lex-parser.shtml#Download) if you do not have already.

# Customization
A couple of things need to be changed.
> in file predict.py
```
path_to_jar = '../../LBS/LBS-X/lib/stanford-parser/stanford-parser.jar'
path_to_models_jar = '../../LBS/LBS-X/lib/stanford-parser/stanford-parser-3.6.0-models.jar'
```
Point to folder where you have downloaded the Stanford Parser

# Running

### parse.py
run only once
```
python parse.py
```
It is implementation of Step 2 in ./doc/approach.

We only need to run this once, because we only need to parse and try to find reminder text to corresponding message text only once. Saving time.
We do classification independently and this helps us save time because Whenever classifier classifies a message text as 1, we just get that reminder text of the corresponding message text.

### train.py
uncomment or comment different functions at the bottom of the file to build learning models.
run it to build models which will be saved in ./models/

### predict.py
either use classify('name') function to independently classify the eval_data.txt
or run 5 classifier in file train.py, and run predict.py to predict best classification using direct voting ensembling.

### clean.py
It is implementation of Step 2 in ./doc/approach.

directly running following will work too.
```
python train.py
python predict.py
python clean.py
```
will make a TSV in current directory named eval_pred_ens.tsv

This TSV is the final classification on the given test data.
This TSV is already present in the repo.
