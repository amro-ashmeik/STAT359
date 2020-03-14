## Summmary

Throughout this quarter, I was responsible for automatic report generation. My goal was to reproduce an automated report generator designed for time series data analysis; similar to that of the current state-of-the-art report generator Automatic Statistician. While Automatic Statistician included model selection and evaluation alongside report generation, my task was focused on report generation. Nevertheless, I analyzed the system used by Automatic Statistician to understand what exactly it does and how it does it. I used the analysis to guide the development of my own report generator.

Report generation lies at the end of the data science pipeline. All of the key results of the model(s) generated need to be consolidated and summarized in an organized manner such that it is presentable to an audience. The purpose of the report should be to help the reader understand the important takeaways through natural language and informative visuals. Depending on the audience, the report should be curated in a way that caters to that audience, such as taking into account how much technical statistics versus how much business application to include.

Report generation is a critical component of the data science pipeline and automating stands to provide many benefits, the most important being time savings. Once a template has been established, producing reports automatically is significantly quicker than manually writing them. With the click of a button, a report can be generated based on the results from some model(s) and this can be done repeatedly for future projects with only slight adjustments for each project. For scientific journals, where format and organization are well defined, an automatic report generator can be extremely useful. In business, automatic report generation saves a business analyst‘s time, overall increasing their productivity. A flexible, robust report generator can have multiple applications throughout a company allowing for a broad increase in productivity across the company.

For the purposes of this project, I developed a report generator based on the following inputs and assumptions:
1.	A .csv file of cleaned data with 2 columns: Date, X (feature X can be any arbitrary numerical value). The name is this file is assumed to be “daily_data.csv”
2.	A .csv file of predictions for feature X fitted on the .csv file from 1.
    -	Multiple .csv files can be exist (e.g. different seeds of a model)
    -	The names of these files are assumed to be “fitted_daily_data_i.csv” where I is the # of the seed
3.	Training period N provided by the user (each prediction point for a date is based on training on N previous days).
4.	A user-inputted modified z-score to use as a threshold. Observations with a modified z-score (based on the median absolute deviation) greater than the provided value will be considered outliers.
5.	A user-inputted confidence that the report generator will use to create a confidence interval for predictions.
6.	A user inputted seed count for the number of seeds/models used and number of fitted data files generated.

My report generator produces a report that includes descriptions of the dataset provided, metrics that describe the forecast accuracy of the model used to make the predictions, and significant dates in the timeseries based on observations that are outliers. The report begins as a template with parameters throughout that will be provided later based on user input or based on descriptions derived. The descriptions I derive from the dataset are:
*	The range of dates covered by the dataset
*	The interval between data points in days
*	The mean of the dataset
*	The median of the dataset

The metrics I used to describe the forecast accuracy of the model are mean absolute error (MAE) and root mean squared error (RMSE). These metrics are easy to interpret and to compute. Interpretability is an extremely important factor in report generators. Results that can automatically be translated into a natural language description or conveyed through visuals are great to offload the time spent interpreting a heavy amount of numerical data.

With regards to significant dates, my report generator provides a table that shows the dates of the significant events with the percent change between the day of the event and the preceding event. The significant events are not outliers of the observations themselves, but outliers in the percent change of some feature X between 2 consecutive days. In addition to the table, the report generator creates a plot of the data, fitted data, and markings at the dates of the significant events. Here is an example of the table and plot that would be generated:

(./example.png)

An example dataset along with respective report generated is included in this repo. 


## Reproducing

## Pipeline and Future Work
