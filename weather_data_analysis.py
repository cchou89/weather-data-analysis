import pandas as pd
import numpy as np
import sys
import glob
import os
import matplotlib.pyplot as plt
import re
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


weather_data = sys.argv[1]
image_data = sys.argv[2]
path = r'{0}'.format(weather_data)
path_im= r'{0}'.format(image_data)

def filename_to_datetime(s):
    pattern = r'^\D*(\d*).jpg'
    prog = re.compile(pattern)
    string = prog.match(s)
    return string[1]

#https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
def file_to_df(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f, skiprows = 16) for f in all_files)
    data = pd.concat(df_from_each_file, ignore_index=True)
    return data

def clean_weather_data(data, images):
    clean_data = data
    clean_data = clean_data.drop(['Hmdx'], axis = 1)
    clean_data['Date/Time'] = pd.to_datetime(clean_data['Date/Time'], infer_datetime_format = True, format='%Y%m%d%h')
    clean_data['Time'] = clean_data['Date/Time'].map(lambda x: x.hour)
    clean_data = clean_data.drop(['Data Quality'], axis = 1)
    clean_data = clean_data.rename(index=str, columns={'Temp (°C)': 'Temp', 
                                                       'Dew Point Temp (°C)': 'Dew Point Temp',
                                                       'Wind Dir (10s def)' : 'Wind Dir',
                                                       'Time': 'Hour'
                                                       })
    clean_data = clean_data.dropna(axis = 1, how = 'all')
    clean_data = images.join(clean_data.set_index('Date/Time'), on= 'Date/Time')
    clean_data = clean_data.dropna()
    clean_data['Weather'] = clean_data['Weather'].apply(classify_weather, args = [r'.*(Cloudy)'])
    clean_data['Weather'] = clean_data['Weather'].apply(classify_weather, args = [r'.*(Clear)'])
    clean_data['Weather'] = clean_data['Weather'].apply(classify_weather, args = [r'.*(Rain|Snow|Fog)'])
    #    clean_data['Weather'] = clean_data['Weather'].apply(classify_rain, args = [r'(Drizzle|Thunderstorms)'])
    clean_data = clean_data.dropna(axis=1, how='all')
    return clean_data
    
def annual_data(data, year):
    return data[data['Year'] == year]

def process_images(path):
    all_images = glob.glob(os.path.join(path, '*.jpg'))
    image_list = pd.DataFrame({'filename': all_images})
    image_list['Date/Time'] = image_list['filename'].apply(filename_to_datetime)
    image_list['Date/Time'] = pd.to_datetime(image_list['Date/Time'], infer_datetime_format = True, format='%Y%m%d%h')
    return image_list

def classify_rain(data, weather):
    compiler = re.compile(weather)
    string = compiler.search(data)
    if string != None:
        return 'Rain'
    else:
        return data
    
def classify_weather(data, weather):
    compiler = re.compile(weather)
    string = compiler.search(data)
    if string != None:
        return string[1]
    else: 
        return data

def scatter_plot(x, y, color):
    plt.figure(figsize=(12, 4))
    plt.plot(x, y, 'b.', alpha=0.2)
    plt.show()

#https://stackoverflow.com/questions/45896800/how-to-convert-image-to-dataset-to-process-machine-learning 
def reshape_to_2d(file_list):
    X= []
    for file in file_list:
        im=Image.open(file)
        imarr=np.array(im)
        flatim=imarr.flatten()
        X.append(flatim)
    return X

def fit_model(X, y):
    model = make_pipeline(StandardScaler(), PCA(75), SVC(kernel='linear', C=2.0))
    model.fit(X, y)
    return model
        
data = file_to_df(path)
images = process_images(path_im)
clean_data = clean_weather_data(data, images)

#check how many labels there are
#unique = list(clean_data['Weather'].unique())
file_list = list(clean_data['filename'])
X = reshape_to_2d(file_list)
y = clean_data['Weather']
y_hour = clean_data['Hour']
y_month = clean_data['Month']

jpg_check = r'^.*(\.jpg)'
checker = re.compile(jpg_check)
test_input = sys.argv[3]
#model = make_pipeline(StandardScaler(), PCA(75), SVC(kernel='linear', C=2.0))

if test_input == 'test':
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = fit_model(X_train, y_train)
    print('Weather score:')
    print(model.score(X_test, y_test))
    df = pd.DataFrame({'True': y_test, 'Predicted': model.predict(X_test)})
    df.to_csv('test')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_hour)
    model = fit_model(X_train, y_train)
    print('Hour score:')
    print(model.score(X_test, y_test))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_month)
    model = fit_model(X_train, y_train)
    print('Month score:')
    print(model.score(X_test, y_test))
    
else:
    if checker.match(test_input) != None:
        input_data = Image.open(test_input)
        imarr=np.array(input_data)
        flatim=imarr.flatten()
        input_t = []
        input_t.append(flatim)
    else:
        image_list = process_images(test_input)
        input_t = reshape_to_2d(image_list)
    model_w = fit_model(X, y)
    model_h = fit_model(X, y_hour)
    model_m = fit_model(X, y_month)
    df = pd.DataFrame({'p_month':model_m.predict(input_t),
                       'p_hour': model_h.predict(input_t),
                       'p_weather': model_w.predict(input_t)})
    df.to_csv('results.csv')