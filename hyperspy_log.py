#!/usr/bin/env python 
# ============================
# 2019-04-03 
# 13:26 
# ============================
quit()
s = tomotools.load('tomotools/tests/test_data/HAADF.mrc')
ali = s.stack_register('ECC')
ali.show()
