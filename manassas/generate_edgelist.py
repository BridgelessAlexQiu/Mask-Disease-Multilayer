import numpy as np
from scipy import sparse

f_read = open("manassas_raw.txt", 'r')
f_write_1 = open("manassas_social.edges", 'w')
f_write_2 = open("manassas_bio.edges", 'w')

# Skip the first two lines that are attributes
next(f_read)
next(f_read)

list_of_name = []

for line in f_read:
    u = line.split(",")[0]
    v = line.split(",")[2]
    list_of_name.append(u)
    list_of_name.append(v)


list_of_name = list(set(list_of_name)) # remove duplicates
name_id_mapping = {} # name : id

for id, name in enumerate(list_of_name):
    name_id_mapping[name] = id

f_read.seek(0)
next(f_read)
next(f_read)

for line in f_read:
    u = line.split(",")[0]
    v = line.split(",")[2]
    t = (line.split(",")[1]).split(':')[1]

    edge = str(name_id_mapping[u]) + " " + str(name_id_mapping[v]) + "\n"
    
    if int(t) != 1: # Not home interaction
        f_write_1.write(edge)

    f_write_2.write(edge)

f_read.close()
f_write_1.close()
f_write_2.close()
