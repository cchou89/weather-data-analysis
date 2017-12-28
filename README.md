A Python ML program to identify weather based off of webcam images using the scikit library

Input: image file (.jpg) or image folder
Outputs: csv of weather labels and/or accuracy scores

example:
weather_data_analysis yvr-weather katkam-scaled pic.jpg
#This produces a csv file with the predictions

weather_data_analysis yvr-weather katkam-scaled test
#This produces a csv file and accuracy scores for months, hours and weather labels

Dependencies:
pandas
numpy
sys
glob
os
matplotlib
re
sklearn
PIL

