import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pdflatex import PDFLaTeX
from datetime import date
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import sys

def mean_confidence_interval(data,confidence=0.95):
    if data.isnull().values.any():
        return np.empty(3)*np.nan
    confidence=0.95
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return [m-h,m+h,m]

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    
    return modified_z_score > thresh

def generate_results(data,fitted_data,z_thresh,training_period,confidence):
    
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
    fitted_data.iloc[:, 0] = pd.to_datetime(fitted_data.iloc[:, 0])
    
    #Finding significant events
    data["change"] = abs(data.iloc[:, 1].pct_change())
    change = np.array(data["change"])[1:]
    outliers = is_outlier(change,thresh=z_thresh)
    significant_dates = data.iloc[:, 0][1:][outliers]
    significant_dates_values = np.round(change[outliers],2)
    
    #Range of dates over the data
    min_date = min(data.iloc[:, 0].apply(pd.Timestamp)).date()
    max_date = max(data.iloc[:, 0].apply(pd.Timestamp)).date()
    
    #Interval of dates (x-tick)
    date_intervals = []
    for i in range(1,len(data)):
        date_intervals.append((data.iloc[i,0] - data.iloc[i-1,0]).days)

    date_interval = np.median(date_intervals)
    
    true = data.iloc[:, 1][training_period+1:]
    fitted_data[['CI_lower','CI_upper', 'CI_mean']] = fitted_data.iloc[:,1:].apply(lambda row: pd.Series(mean_confidence_interval(row, confidence=confidence)), axis=1)
    fitted_data.to_csv("{}_Prediction_Confidence_Intervals.csv".format(dataset_name), index=False)
    pred = fitted_data["CI_mean"][training_period+1:]
    
    #Forecast Accuracy
    MAE = round(mean_absolute_error(true, pred), 2)
    RMSE = round(mean_squared_error(true, pred, squared=False), 2)
    
    #Generating plots
    data.iloc[:, 0] = data.iloc[:, 0].apply(pd.Timestamp)
    fitted_data.iloc[:, 0] = fitted_data.iloc[:, 0].apply(pd.Timestamp)
    
    #Significant Event Plots
    fig,ax = plt.subplots()

    plt.locator_params(axis='x', nbins=5)


    ax.vlines(significant_dates, ymin=0, ymax=max(data.iloc[:, 1]) + 10 ,color='r', linestyles="dashed")

    true, = ax.plot(data.iloc[:, 0], data.iloc[:, 1], label="True")
    fitted, = ax.plot(fitted_data.iloc[:, 0], fitted_data.iloc[:, 1], label="Fitted")
    plt.legend([true, fitted], ['True', 'Fitted'])
    ax.xaxis_date()
    
    column_names = data.columns
    x_label = column_names[0]
    y_label = column_names[1]
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("{} vs {}".format(x_label, y_label))

    fig.savefig("sig_events.png")
    
    #Basic timeseries plot.
    fig,ax = plt.subplots()

    plt.locator_params(axis='x', nbins=5)
    
    true, = ax.plot(data.iloc[:, 0], data.iloc[:, 1], label="True")
    fitted, = ax.plot(fitted_data.iloc[:, 0], fitted_data.iloc[:, 1], label="Fitted")
    plt.legend([true, fitted], ['True', 'Fitted'])
    ax.xaxis_date()
    
    column_names = data.columns
    x_label = column_names[0]
    y_label = column_names[1]
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("{} vs {}".format(x_label, y_label))

    fig.savefig("basic_plot.png")
    
    
    return significant_dates, significant_dates_values, min_date, max_date, date_interval, MAE, RMSE




def generate_report(title,author,dataset_name,data,fitted,z_thresh,training_period,confidence):

    significant_dates, significant_dates_values, min_date, max_date, date_interval, MAE, RMSE = generate_results(data,fitted,z_thresh,training_period,confidence)

    significant_dates_str = ",".join(str(date.date()) for date in significant_dates)

    table = "\\hline\\hline "

    for date,value in zip(significant_dates,significant_dates_values):
        table += "{date} & {value} \\\ ".format(date=date.date(),value=value)

    content = f'''\documentclass {{article}}
    \\usepackage {{graphicx}}
    \\usepackage[parfill]{{parskip}}
    \\begin {{document}}
    \\title {{{title}}}
    \\author {{{author}}}
    \date {{{date.today().date()}}}
    \maketitle

    \section {{Dataset Summary}}

    The dataset {dataset_name} has data covering the range of dates between {min_date} and
    {max_date}. The interval between data points is {date_interval} days. Forecasts are calculated
    based on training the model on the {training_period} days preceding the date being forecasted. The
    mean and median of {data.columns[1]} are {round(np.mean(data.iloc[:,1]),2)} and {round(np.median(data.iloc[:,1]),2)}
    respectively. Below is the timeseries plot of the data.
    
    \\begin {{center}}
    \\includegraphics[scale=0.8]{{basic_plot.png}}
    \end {{center}}

    \section {{Forecast Accuracy}}

    The forecast accuracy of the model was evaluated using 2 metrics: mean absolute error (MAE) and root
    mean squared error (RMSE).

    MAE = {MAE}

    RMSE = {RMSE}

    \section {{Significant Events}}

    The following signficant event dates were captured using a z-score threshold of {z_thresh}:
    
    \\begin{{center}}
     \\begin{{tabular}}{{||c c||}} 
     \hline
     {data.columns[0]} & {"pctChangeIn" + data.columns[1]} \\\ [0.5ex] 
     {table} [1ex] 
     \hline
    \end{{tabular}}
    \end{{center}}
    
    The following plot shows the timeseries data along with red dashed lines 
    representing the dates of the significant events.

    \\begin {{center}}
    \\includegraphics[scale=0.8]{{sig_events.png}}
    \end {{center}}

    \end {{document}}
    '''



    with open('{} Report.tex'.format(dataset_name),'w') as f:
        f.write(content)

    f.close()

    os.system("pdflatex '{} Report.tex'".format(dataset_name))



print(sys.argv)
title = sys.argv[1]
author = sys.argv[2]
dataset_name = sys.argv[3]
z_thresh = float(sys.argv[4])
training_period = int(sys.argv[5])
confidence = float(sys.argv[6])
seeds = int(sys.argv[7])

fitted = pd.read_csv("fitted_daily_data_0.csv")
for i in range(1,seeds):
    
    d = pd.read_csv("fitted_daily_data_{}.csv".format(i)).iloc[:,1]
    fitted[i] = d
        

fitted.columns = range(fitted.shape[1])
df = pd.read_csv("daily_data.csv")

generate_report(title=title, author=author, dataset_name=dataset_name, data=df, fitted=fitted, z_thresh=z_thresh, training_period=training_period, confidence=confidence)





