import glob
import csv
import os
import pprint

def get_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)                      
        result = {tuple(row[:2]): row[2] for row in reader}
        return result   

all_data=[]
result = {}
path='C:\\Users\\david\\Desktop\\Base Method Predictions\\Ecoli-noise0.0-nodes200' 

for infile in glob.glob(os.path.join(path, '*.csv')):

    print ("Current File Being Processed is:  " + infile) 
    (PATH, FILENAME) = os.path.split(infile)
    all_data.append(get_data(infile))

i= 0
for dic in all_data:
    i+=1 
    for key in ( dic.keys() | result.keys()):
        if key in dic:           
            if key in result:
                result.setdefault(key, []).append(dic[key])
                
            else:
                    x = 1
                    while x < i:
                        result.setdefault(key, []).append(0)
                        x+=1
                    result.setdefault(key, []).append(dic[key])        

        else: result.setdefault(key, []).append(0)

with open('C:\\Users\\david\\Desktop\\Ecoli-noise0.0-nodes200.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for date in result:
       writer.writerow([date] + result[date])
