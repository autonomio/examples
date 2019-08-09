import talos

# load the breast cancer dataset
x, y = talos.templates.datasets.breast_cancer()

# automatically load a broad parameter space
param = talos.autom8.AutoParams(network=False)

# tweak the parameter space slightly
param.batch_size(12, 48, 8)
param.neurons(10, 50, 10)
param.lr([.5, 0.75, 1, 1.25, 1.5])
param.layers(0, 6, 1)
param.epochs(50, 250, 50)

# setup an auto experiment
auto = talos.autom8.AutoScan(task='binary',
                             experiment_name='breast_cancer')

# let the good times roll
breast_cancer = auto.start(x, y,
                           params=param.params,
                           round_limit=5000,
                           reduction_method='gamify')
