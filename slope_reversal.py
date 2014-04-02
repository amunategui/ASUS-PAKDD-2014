import pandas as pd
import numpy as np
from pandas.stats.moments import ewma
from sklearn.feature_extraction import DictVectorizer
import time
from collections import defaultdict
import difflib
from datetime import date
from sklearn.feature_extraction import DictVectorizer
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from dateutil import relativedelta
import math
def buildyearladder():
    timelist = []
    for year in range(2005,2010):
        for month in range(1,13):
            timekey = str(year) + "," + str(month) #  "{0:0=2d}".format(month)
            timelist.append(timekey)
    return timelist
def firsttransform(df):
    history = []
    history_keys = []
    timeladder = buildyearladder()
  
    masterdict = defaultdict(int)
    for m in range(1,10):
        for p in range(1,32):
            for t in timeladder:
                masterdict[str(m) + "," + str(p) + "," + t] = 0
    keywithvalues = defaultdict(int)
    for count, row in df.iterrows():
        trepair = row['year/month(repair)']
        module = int((row['module_category']).replace("M",""))
        component = int((row['component_category']).replace("P",""))
        year_repair = int(trepair.split('/')[0])
        month_repair = int(trepair.split('/')[1])
        thiskey = str(module) + "," + str(component) + "," + str(year_repair) + "," + str(month_repair)
        if thiskey in masterdict:
            masterdict[thiskey] += int(row['number_repair'])
    completelist = []
    for k,v in masterdict.items():
        module = int(k.split(",")[0])
        component = int(k.split(",")[1])
        year_repair = int(k.split(",")[2])
        month_repair = int(k.split(",")[3])
        completelist.append([module,component,year_repair,month_repair,v])
    df = pd.DataFrame(completelist, columns=['module', 'component', 'year_repair', 'month_repair', 'repair'])
    df = df.sort_index(by=['module', 'component', 'year_repair', 'month_repair'])
   
    newkey = ''
    counter = 0
    completelist_counter = []
    for count, row in df.iterrows():
        module = int(row['module'])
        component = int(row['component'])
        year_repair = int(row['year_repair'])
        month_repair = int(row['month_repair'])
        repair = int(row['repair'])
        counter += 1
        if newkey != str(module) + "," + str(component):
            counter = 1
        completelist_counter.append([module,component,counter,repair])
        newkey = str(module) + "," + str(component)
  
    df = pd.DataFrame(completelist_counter, columns=['module', 'component', 'month', 'repair'])
    df = df.sort_index(by=['module', 'component', 'month'])
    return df
    # train[(train.module==9) & (train.component==5)].head(100)
 
train_csv = pd.read_csv('RepairTrain.csv')
output_target = pd.read_csv('Output_TargetID_Mapping.csv')
submission = pd.read_csv('SampleSubmission.csv')
 
train = firsttransform(train_csv)



# build list from peak on down

from sklearn import linear_model
clf = linear_model.LinearRegression()
valstosubmit = []
last_module_component = ""
index_count = 0
counter = 0
transformers = defaultdict(list)
transformer_modules = defaultdict(list)
finalvalueinlist = -1

for i in range(0, output_target.shape[0]):
    pred = 0
    module = int((output_target['module_category'][i]).replace("M",""))
    component = int((output_target['component_category'][i]).replace("P",""))
    if last_module_component != str(module) + "," + str(component):
        counter = 61
        index_count = 1
        x = list(train[(train.module==module) & (train.component==component)]['repair'])
        finalvalueinlist = x[-1]
    else:
        counter += 1
        index_count += 1


    if finalvalueinlist <= 0:
        pred = 0
    else:


        lastzerofromstartindex = 0
        bestvaluefromstartindex = 0
        counttillmatch = 0
        # find closest match on begining rung
        
        for p in x:
            if (p > 0 and lastzerofromstartindex == 0):
                lastzerofromstartindex = counttillmatch

            if (finalvalueinlist < p):
                bestvaluefromstartindex = counttillmatch
                break
            counttillmatch += 1

        index_to_reverse = x[lastzerofromstartindex:bestvaluefromstartindex]
        index_to_reverse.reverse()

        guess = index_to_reverse + ([0] * 19)
        pred = guess[index_count-1]
 
            
    valstosubmit.append([i+1, int(np.floor(pred) if ((pred)-0) >= 0 else 0)])
    last_module_component = str(module) + "," + str(component)

 
   
#print(transformers)
predictions = pd.DataFrame(valstosubmit,columns=['id','target'])
predictions.to_csv('submit.csv', sep=',', header=True, index=False)
 