y as np
import pandas as pd
from autonomio.commands import train
%matplotlib inline

## Table of Contents 
## -----------------
## 1. Load data
## 2. Transform data
## 3. Train the model 
## ------------------

## 1. Loading the data from https://www.kaggle.com/c/titanic/data

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

## 2. Transforming and cleaning the data

train_df['Male'] = train_df.Sex == 'male'
test_df['Male'] = test_df.Sex == 'male'
train_df['Female'] = train_df.Sex == 'female'
test_df['Female'] = test_df.Sex == 'female'

train_df['Q'] = train_df.Embarked == 'Q'
train_df['S'] = train_df.Embarked == 'S'
train_df['C'] = train_df.Embarked == 'C'

test_df['Q'] = test_df.Embarked == 'Q'
test_df['S'] = test_df.Embarked == 'S'
test_df['C'] = test_df.Embarked == 'C'

train_df['Cabin_A'] = train_df.Cabin.str.startswith('A') == True
train_df['Cabin_B'] = train_df.Cabin.str.startswith('B') == True
train_df['Cabin_C'] = train_df.Cabin.str.startswith('C') == True
train_df['Cabin_D'] = train_df.Cabin.str.startswith('D') == True
train_df['Cabin_E'] = train_df.Cabin.str.startswith('E') == True
train_df['Cabin_F'] = train_df.Cabin.str.startswith('F') == True
train_df['Cabin_G'] = train_df.Cabin.str.startswith('G') == True
train_df['Cabin_T'] = train_df.Cabin.str.startswith('T') == True
train_df['Cabin_NaN'] = train_df.Cabin.str.startswith('NaN') != False

test_df['Cabin_A'] = test_df.Cabin.str.startswith('A') == True
test_df['Cabin_B'] = test_df.Cabin.str.startswith('B') == True
test_df['Cabin_C'] = test_df.Cabin.str.startswith('C') == True
test_df['Cabin_D'] = test_df.Cabin.str.startswith('D') == True
test_df['Cabin_E'] = test_df.Cabin.str.startswith('E') == True
test_df['Cabin_F'] = test_df.Cabin.str.startswith('F') == True
test_df['Cabin_G'] = test_df.Cabin.str.startswith('G') == True
test_df['Cabin_T'] = test_df.Cabin.str.startswith('T') == True
test_df['Cabin_NaN'] = test_df.Cabin.str.startswith('NaN') != False

train_df = train_df.drop(['Ticket','Cabin','Embarked','Sex'],axis=1)
test_df = test_df.drop(['Ticket','Cabin','Embarked','Sex'],axis=1)

train_df = train_df.dropna()
test_df = test_df.dropna()

## 3. Training the model with Autonomio

train([2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],'Survived',train_df,
                                dims=8,
                                flatten='none',
                                epoch=250,
                                dropout=0,
                                batch_size=12,
                                loss='logcosh',
                                activation='elu',
                                layers=6,
                                shape='brick',
                                save_model='titanic')
