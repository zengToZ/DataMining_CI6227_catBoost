import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate attributes(column):
# a0=audience count (k)	
# a1=showing time (min)	
# a2=Type (code 1-8)	
# a3=Budget (millions)	
# a4=director previous works (0-max)	
# a5=main characters (1-10)	
# a6=Theatre occupancy ratio	
# a7=Movie title length (1-100)	
# a8=production company

a0,a1,a2,a3,a4,a5,a6,a7,a8,labels = [],[],[],[],[],[],[],[],[],[]
size = 15000

for i in range(size):
    a0.append(np.random.randint(10,1000))
    a2.append(np.random.randint(1,9))
    a3.append(np.random.random_sample()*299.9+0.1)
    a4.append(np.random.randint(0,21))
    a5.append(np.random.randint(1,11))
    a6.append(np.random.random_sample()*0.99+0.01)
    a7.append(np.random.randint(1,101))
    a8.append(np.random.randint(1,9))
    labels.append(-1)
# Use beta distribution to set showing time (around 90mins)
a1 = np.random.beta(2,5,[1,size])
a1 = np.rint(a1[0]*120+60)
# Plot distribution of attribute 'showing time'
plt.hist(a1,100)
plt.show()

# Generate Labels:
# Apply F(a0,a1,..,a8)=  (a0/1000 - abs(a1-90)/600+a2^0.5/5+ b(a3) + d(a4) + (10-a5)*0.001)*5*a6^0.2, to set labels
for i in range(size):
    # Budget fucntion:
    if a3[i] < 1: 
        b = 0.1
    elif a3[i] >=1 and a3[i] < 5:
        b = 0.125
    elif a3[i] >=5 and a3[i] < 20:
        b = 0.15
    elif a3[i] >=20 and a3[i] < 100:
        b = 0.1275
    elif a3[i] >=100:
        b = 0.2
    # Director previous work function:
    if a4[i] < 3:
        d = 0.1
    elif a4[i] >=3 and a4[i] < 10:
        d = 0.2
    elif a4[i] >= 10:
        d = 0.3  
    # Set Labels    
    labels[i] = np.rint((a0[i]/1000 - np.abs(a1[i]-90)/600 + np.power(a2[i],0.5)/5 + b + d + (10-a5[i])*0.001)*5*np.power(a6[i],0.2)) 

# Assemble the dataset
df = pd.DataFrame({'a0':a0,'a1':a1,'a2':a2,'a3':a3,'a4':a4,'a5':a5,'a6':a6,'a7':a7,'a8':a8,'labels':labels})
print(df.shape)
print(df.dtypes)

# Set Labels: 
#1=very boring
#2=boring
#3=so-so
#4=good
#5=very good
for i in range(size):
    # Convert to 1 - 5 labels:
    if df['labels'][i] < 2: 
        df.iat[i,9] = 1
    elif df['labels'][i] >= 2 and df['labels'][i] < 4:
        df.iat[i,9] = 2
    elif df['labels'][i] >= 4 and df['labels'][i] < 6:
        df.iat[i,9] = 3
    elif df['labels'][i] >= 6 and df['labels'][i] < 8:
        df.iat[i,9] = 4
    elif df['labels'][i] >= 8:
        df.iat[i,9] = 5

# Generate CSV file
df.to_csv("movie_rating.csv",index=False)
