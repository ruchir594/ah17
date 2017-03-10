# Find any reminder text from message text

### To understand the approach, head to [docs/approach.rtf](https://github.com/ruchir594/ah17/blob/master/docs/approach.rtf) in this repo.

## Adding files
Make a directory "data" in this repo.
Add following files...
> training_data.tsv
> eval_data.tsv

run
```
python convert.py
```
Also run
```
pip install -r requirements.txt
```
this will make a new TSV training_impro.tsv in ./data repository

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
run only once
```
python parse.py
```
this is Entire Step 2 in ./doc/approach.
We only need to run this once, because we only need to parse and try to find reminder text to corresponsing message text only once. Saving time.
We do classification independently and this helps us save time because Whenever classifier classifies a message text as 1, we just get that reminder text of the corresponding message text.

open train.py file.
uncomment or comment different functions at the bottom of the file to build learning models.
run it to build models which will be saved in ./models/

open predict.py file.
either use classify('name') function to independently classify the eval_data.txt
or run 5 classifier in file train.py, and run predict.py to predict best classification using direct voting ensembling.

then run clean.py.

running
```
python train.py
python predict.py
python clean.py
```
will make a TSV in current directory named eval_pred_ens.tsv
