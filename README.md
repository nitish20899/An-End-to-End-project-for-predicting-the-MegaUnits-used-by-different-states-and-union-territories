# End to End project for predicting Electricity ( MegaUnits ) used by different states and union territories in India using univariante LSTM model. 

Please use the below link to see the project file (ipynb) if you face difficulty in loading the notebooks.

* link of the notebook which contains End to End process (note: As this ipynb file is 93.5 mb I had created a pdf version of it .The link is below. original ipynb file is also available in my repository)
https://nbviewer.jupyter.org/github/nitish20899/An-End-to-End-project-for-predicting-the-Megaunits-used-by-different-states-and-union-territories/blob/master/End_to_End_notebook1.pdf

* link of the notebook which contains the way i scrapped the data https://nbviewer.jupyter.org/github/nitish20899/An-End-to-End-project-for-predicting-the-Megaunits-used-by-different-states-and-union-territories/blob/master/scrapping_data_notebook.ipynb

* deployed link https://forecast-1111.df.r.appspot.com/



## Table of Contents
* [Introduction](#Introduction)
* [Aim of the project](#Aim)
* [About the Dataset](#Dataset)
* [Process in detail](#Process)
* [All models accuracies](#Accuracy)
* [Final results](#Results)
* [Legality](#Legality)



## Introduction

As a data science enthusiast, I was a type of person who likes applying knowledge on the real world data. While i was on search of a data, i came across a website https://posoco.in/. Power System Operation Corporation Limited is a wholly owned Government of India enterprise under the Ministry of Power. 

It is responsible to ensure the integrated operation of the Grid in a reliable, efficient, and secure manner.This website uploads a PDF every week which contains the number of Mega units used by every state. 

They are uploading the data from the past 2013. So I decided to create a model that helps in predicting the number of Mega units used by the different states in the future using the Univariate LSTM model.

In order to get the data from 2013 to 2020, I need to scrap the data from  420+ PDF's. Some of the data is in image format too. Not only scrapping the data I had to clean it, Explore the data, preprocess it, create a model, created a webpage, and deploy it using Google Cloud Platform.

I felt so excited and satisfying while doing this project and learned many things like , methods to preprocess the time series data, tuning the models with methods like Bayesian optimization using keras-tuner ...etc.



## Aim 

The main aim of this end to end project is to scrap the data from multiple PDF and images and use that data for creating univariant LSTM models for each state and union teritories for predicting the future usage of Mega units used by them.



## Dataset

After scraping, properly arranging and cleaning the data, the dataframe has 37 columns adn 2593 rows.

All the column values are float except date column. Columns represent 29 states and 7 union teritories of india.

For the columns Andra pradesh and telangana I am going to take data from the year 2015 ( As AP and TL has been divided on 2014 
july ,the data before that is inaccurate, so considering from 2015 will give us accurate data )

For other columns i am going to take the data from 2013.

For task evaluation,I considered from 2013-01-07 to 2019-07-01 as Training Set, and Test Set is from 2019-07-02 to 2014-11-14. This is roughly a 80%/20% split.

For training i took 48 datapoints as X_train and 49th one as y_train ( y label ) incrementally for the whole training set. 



## Process

* Scrapping the data
   * I wrote a code which downloads all the weekly data pdf's from this link https://posoco.in/reports/weekly-reports/
   * For scrapping i used beautifulsoup ,bs4 ,requests and other required libraries.
   * There are 434 pdf's present in the website.
   * I got the required table from the pdf's.
   * There are some situations where , the table I want is in an image format so i used ExtractTable library.
   * Finally merged all the tables from different pdf's.
* Data cleaning
   * Cleaned the Date column
   * Date column is converted into timestamp
   * Dropped duplicate rows
   * Dealed with null values
* Exploratory Data Analysis
  * visualized the total number of megaunits used by different states using folium library
    <p align= "center">
        <img width="800" height="490" src="https://user-images.githubusercontent.com/63724986/104691733-cfb73500-572c-11eb-9e3a-80b9e569dad0.jpg"> 
    </p>
  * Visualizing the distribution of data and planning about preprocessing
    <p align= "center">
        <img width= "800" height="490" src="https://user-images.githubusercontent.com/63724986/104698169-f4180f00-5736-11eb-845a-96d55492b881.jpg">
     </p>   
  * Visualizing the data whether it is stationary or not by ploting rolling mean and rolling std
     <p align= "center">
         <img width= "800" height="290" src="https://user-images.githubusercontent.com/63724986/104698526-7c96af80-5737-11eb-85fd-29852bd1c0b9.png">
     </p>   
  * As you can see the data is not stationary  
* Data Preprocessing
   * Used zscore to remove outliers
   * Used Augumented Dicky Fuller test to find which columns are not stationary 
   * Applied seasonal decompose to see the trend and seasonality of the data separately
   * We are going to tranform the non-stationary data to stationary using differencing transform.
   * After applying differencing transform the Andhra Pradesh column look like below
<p align= "center">
   <img width= "800" height="290" src="https://user-images.githubusercontent.com/63724986/104723036-a6aa9a80-5754-11eb-8fb1-80615cb01be6.png">
</p> 

* Creating Univariant LSTM model With hyperparameter tuning
   * First it loads the dataframe from the csv file I saved
   * It removes first 441 rows , because I am going to take data from the year 2015 ( As AP and TL has been divided on 2014 july,the data before that is inaccurate,so considering from 2015 will give us accurate data )
   * creating a function to differencing transform the data 
   * Now , taking first 38 rows for training and 39th row as a test data ,incrementally for the whole data
   * converting training data and test data to numpy array and the dtype is float ( dtype=float32 because tensorflow works on float32 values in default, float64 values may cause warnings and errors)
   * Dividing the data for training and testing.
   * last 180 rows is for testing and the remaining is for training the data
   * Now I am reshaping the training data to 3d
   * I had created a LSTM model using BayesianOptimization from keras-tuner
   * BayesianOptimization tuner helps us find the best combination of layers, neurons and value within the given range
   * after creating the model , I am going to get the best model from all the trails and use it to test the data
   * After getting the accuracy on the test data , I will plot it
   * I am going to save the model as column name_model_75.h5
   * Now I will train the model with 100% data for better future predictions and I will save it as column name_model_100.h5
   * Later I used that saved model to make future predictions
* Predicting future by using the models we trained
   * For predicting the future we are going to train the same models with 100% data for better future predictions
* Creating the required codes for deployment (GCP)



## Accuracy

* 36 models were created for 36 states and union territories
* Accuracy of each model is ranging between 86.3 to 98.5 on test data

| States_models     |   Accuracy |
|:------------------|-----------:|
| andhra pradesh    |       98.5 |
| arunachal pradesh |       93.9 |
| assam             |       94.4 |
| bihar             |       93.8 |
| chandigarh        |       93.5 |
| chhattisgarh      |       95.9 |
| dd                |       93.9 |
| delhi             |       94.4 |
| dnh               |       96.9 |
| dvc               |       97.6 |
| essar steel       |       89.3 |
| goa               |       94.8 |
| gujarat           |       96.9 |
| haryana           |       94.5 |
| hp                |       95.4 |
| j&k               |       94.8 |
| jharkhand         |       95.8 |
| karnataka         |       95.9 |
| kerala            |       96.8 |
| maharashtra       |       97.4 |
| manipur           |       94.4 |
| meghalaya         |       95.5 |
| mizoram           |       94.6 |
| mp                |       97   |
| nagaland          |       95.7 |
| odisha            |       95.1 |
| pondy             |       95.1 |
| punjab            |       93.6 |
| rajasthan         |       96.7 |
| sikkim            |       86.3 |
| tamil nadu        |       96.3 |
| telangana         |       97.5 |
| tripura           |       91.6 |
| up                |       95.3 |
| uttarakhand       |       94.4 |
| west bengal       |       94.3 |



## Results


I had successfully deployed my project through Google Cloud Platform. you can click the link below to see the deployed project
https://forecast-1111.df.r.appspot.com/

Below are the screenshots of the website
<p align= "center">
   <img width= "800" height="400" src="https://user-images.githubusercontent.com/63724986/104751771-d6b96400-577b-11eb-9d7b-2cf885df45e8.jpg">
</p> 
<p align= "center">
   <img width= "800" height="400" src="https://user-images.githubusercontent.com/63724986/104751775-d8832780-577b-11eb-9879-6b4b3fe3a8ce.jpg">
</p> 



## Legality

This is a personal project made for non-commercial uses ONLY. This project will not be used to generate any promotional or monetary value for me, the creator, or the user.
I personally thank the POSOCO team for keeping the data open.

### Thank you for looking at my project. Any queries or Suggestions please mail at nitishkumar2902@gmail.com
