import dedomena as da
import numpy as np

#load icu burden dataset
data=da.datasets.autonomio('icu_burden')
#drop unnecessary columns and fill nan values with 0
data=data.drop(['id','month_survival','year_survival','age_group_range'],axis=1).fillna(0)

#use visualisation libraries to see relationships between columns.Here we draw a line chart between age and icu duration
import astetik as ast
ast.line(data, x='age_in_years',y='icu_duration_hours',title='ICU burden data', sub_title="survival rate of icu patients")

#use data transformation library to detect the task from the target column and train a model
from wrangle import array
y_type, y_range, y_format=array.array_detect_task(data["hospital_survival"])


   
import talos

# convert the the icu_burden dataset into numpy array with hospital survival as target and remaining columns as x
x=np.asarray(data.loc[:, data.columns!='hospital_survival'])
y=np.asarray(data["hospital_survival"])

# automatically load a broad parameter space
param = talos.autom8.AutoParams(network=False)

# tweak the parameter space slightly
param.batch_size(12, 48, 8)
param.neurons(10, 50, 10)
param.lr([.5, 0.75, 1, 1.25, 1.5])
param.layers(0, 6, 1)
param.epochs(50, 250, 50)

params=param.params
# setup an auto experiment
auto = talos.autom8.AutoScan(task=y_type,
                              experiment_name='icu_burden')

# let the good times roll
icu_burden = auto.start(x, y,
                            params=params,
                            round_limit=5000,
                            reduction_method='gamify')






