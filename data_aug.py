import pandas as pd
from sklearn.preprocessing import LabelEncoder

# data preprocessing

def legend(csv_file):
	df = pd.read_csv(csv_file)

	df['emotion'] = df['emotion'].apply(lambda x: x.lower())
	#encoding label data to unique int
	le = LabelEncoder()
	df['emotion'] = le.fit_transform(df['emotion'])

	return df