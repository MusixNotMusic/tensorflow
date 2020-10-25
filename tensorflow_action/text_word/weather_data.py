import os

data_dir = '/Users/chengjinpeng/Documents/deeplearning_data_sets'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv');

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))