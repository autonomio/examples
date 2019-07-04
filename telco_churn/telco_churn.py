def data_prep():

	import sys

	for i in ['wrangle']:
	    sys.path.insert(0, '/Users/mikko/Documents/GitHub/' + i)


	import wrangle
	import pandas as pd
	import numpy as np

	df = pd.read_csv('telco-customer-churn.zip')

	df = wrangle.df_fill_empty(df, np.nan)
	df = wrangle.df_drop_nanrows(df)
	df = wrangle.df_to_multilabel(df, max_uniques=4, ignore_y='Churn')
	df = wrangle.df_rename_cols(df)
	df.drop('C0', 1, inplace=True)
	df = wrangle.col_move_place(df, 'C4', 'last')
	df = wrangle.df_rename_col(df, 'C4', 'Y')
	df = wrangle.col_to_biclass(df, 'Y', 'Yes')
	df = wrangle.df_to_numeric(df)
	df[['C1', 'C2', 'C3']] = wrangle.df_rescale_meanzero(df[['C1', 'C2', 'C3']])
	x, y = wrangle.df_to_xy(df, 'Y')

	return x, y