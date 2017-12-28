A Python ML program to identify weather based off of webcam images using the scikit library

This script uses the following to train it's machine learning model:

		Historical climate data found gathered by the government of Canada:
		http://climate.weather.gc.ca/historical_data/search_historic_data_e.html

		Weather images used as part of this program was obtained from KatKam images:
		www.katkam.ca

Input: data_folder, image_folder, image_file
Outputs: csv of weather labels and/or accuracy scores

example:
weather_data_analysis data_folder, image_folder, image_file.jpg
#This produces a csv file with the predictions

weather_data_analysis data_folder, image_folder, image_file.jpg
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

