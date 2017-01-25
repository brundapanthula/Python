import pandas as pd
import numpy as np

def __init__(self,datapath):
    path = datapath

def read_data(self,path):
    train = pd.read_csv(path + "\\train.csv") # the train dataset is now a Pandas DataFrame
    test = pd.read_csv(path + "\\test.csv") # the train dataset is now a Pandas DataFrame
    # set features and labels (removing Id from features)
    df_train, df_target = train.iloc[:,1:-1], train['SalePrice']
    df_test = test.iloc[:,1:]
    df_allData = pd.concat([df_train, df_test])


def data_insights(self,df_allData):
    # Let's have a peek of the train data
    df_allData.head()
    # More deeper look into data
    # size of data
    instance_count,attr_count = df_allData.shape
    print("number of instances: " ,instance_count)
    print("number of attributes: ",attr_count)
    # Explore attributes
    df_allData.columns
    df_allData.info()
    # check how the values of attributes are distributed
    df_allData.describe()

def handle_missing_values(self,df_allData):
    # Check for missing values
    pd.isnull(df_allData).any()
    # Count missing values
    missing_count = df_allData.isnull().sum().sort_values(ascending=False)
    # there are total 1460 instances.So we can discard the attributes with missing values greater than 30%
    total_count = df_allData.count()
    # List the columns with more than 30 % missing values
    missing_over60 = [n for n in total_count if n < 0.3 * df_allData.shape[0]]
    drop_col =[]
    for r in missing_over60:
        col_drop = total_count[total_count == r].index[0]
        drop_col.append(col_drop)
    print drop_col
    # drop the remove_col
    df_train = df_allData.drop(drop_col,1)
    # Fill the remaining missing value of numerical attributes with mean and non numerical attributes with frequently occuring values
    df_data_new = df_allData
    df_attr = df_data_new.columns
    for attr in df_attr:
        if(df_data_new[attr].dtype == np.dtype('O')):
            df_data_new[attr] = df_data_new[attr].fillna(df_data_new[attr].value_counts().index[0])
        else:
            df_data_new[attr] = df_data_new[attr].fillna(df_data_new[attr].mean())

    # cross-check for the presence of null values
    print(df_data_new.isnull().any().value_counts())

def data_split(self,df_data_new):
    num_attr = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num_data = df_data_new.select_dtypes(include=num_attr)
    df_num_data.shape

    nonnum_attr = ['object']
    df_nonnum_data = df_data_new.select_dtypes(include=nonnum_attr)
    df_nonnum_data.shape

def check_skew(self,df_num_data):
    # check for skewness in data
    df_num_data.skew()
    # Lets plot to get a better understanding
    import matplotlib.pyplot as plt
    import seaborn as sdv
    attr = df_num_data.columns
    for a in attr:
        sdv.violinplot(df_num_data[a])
        plt.xlabel(a)
        plt.show()
    # Correct the skewness
    # log1p function applies log(1+x) to all elements of the column
    df_skew = df_num_data.skew()

    skewed_attr = [s for s in df_skew if(s > 5.0)]
    skewed_attr
    for skf in skewed_attr:
        sk = df_skew[df_skew == skf].index[0]
        df_num_data[sk] = np.log1p(df_num_data[sk])

def check_corr(self,df_num_data,attr):
    # Understand the correlation between features
    # As seen from above plot features like FullBath, GrLivArea, 1stFlrSF are positively correlated with the sale price.There are features that are negatively correlated also
    # As there are many features let us try to remove some features that are least correlated with sales price.
    # Check the correlation
    # since Correlation requires continous data,categorical data can be ignored
    df_corr = df_num_data.corr()

    # Set threshold to select only highly correlated attributes
    threshold = 0.5

    # List of pairs along with correlation above threshold
    df_corr_list = []

    size = df_num_data.shape[1]

    # Check for the highly correlated pairs of attributes
    for i in range(0,size): # for 'size' features
        for j in range(i+1,size): # avoid repetition
            if (df_corr.iloc[i,j] >= threshold and df_corr.iloc[i,j] < 1) or (df_corr.iloc[i,j] < 0 and df_corr.iloc[i,j] <= -threshold):
                df_corr_list.append([df_corr.iloc[i,j],i,j]) # store correlation and columns index

    # Sort to show higher ones first
    sorted_corr_list = sorted(df_corr_list,key=lambda x: -abs(x[0]))

    # Print correlations and column names
    for v,i,j in sorted_corr_list:
        print ("%s and %s = %.2f" % (attr[i],attr[j],v))


    # Here we can see that GarageCars and GarageArea are highly correlated. So we can remove garagecars first
    df_num_data = df_num_data.drop('GarageCars', axis = 1)

def impute_cat_attr(self,df_nonnum_data):
    #Lets explore the categorical features
    #Findout the unique values of each categorical attribute and their counts

    cat_attr = df_nonnum_data.columns
    split = df_nonnum_data.shape[1]
    cat_label = []
    for i in range(0,split):
        cat_train = df_nonnum_data[cat_attr[i]].unique()
        cat_label.append(list(set(cat_train)))

    #We will use onhot encoding to convert to numeric atributes
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    #One hot encode all categorical attributes
    cat_encode = []
    for i in range(0, split):
        #Label encode
        cat_label_encoder = LabelEncoder()
        cat_label_encoder.fit(cat_label[i])
        cat_feature = cat_label_encoder.transform(df_nonnum_data.iloc[:,i])
        cat_feature = cat_feature.reshape(df_nonnum_data.shape[0], 1)
        #One hot encode
        cat_onehot_encoder = OneHotEncoder(sparse=False,n_values=len(cat_label[i]))
        cat_feature = cat_onehot_encoder.fit_transform(cat_feature)
        cat_encode.append(cat_feature)

    # Make a 2D array from a list of 1D arrays
    import numpy
    cat_encoded = numpy.column_stack(cat_encode)

    # Print the shape of the encoded data
    print(cat_encoded.shape)

    #Concat numeric and nonnumeric data
    df_encoded = numpy.concatenate((cat_encoded,df_num_data.values),axis=1)
    df_encoded.shape

model = build_model("H:\\Kaggle\\housingprice")
model.read_data()
model.data_insights()
model.handle_missing_values()
model.data_split()
model.check_skew()
model.check_corr()
model.impute_cat_attr()

















