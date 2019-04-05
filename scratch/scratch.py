#%%
%matplotlib qt
%load_ext autoreload
%autoreload 2
import tomotools.api as tomotools
import matplotlib.pylab as plt
from tomotools import align
#%%
s = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
#%%
s.plot()
#%%
ali = s.stack_register('ECC')
ali.show()



#%%
