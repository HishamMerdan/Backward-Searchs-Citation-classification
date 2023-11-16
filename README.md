# Backward-Searchs-Citation-classification
# Description

This project focuses on the classification of citations in academic papers. It involves processing XML data, extracting features, and building a classifier to categorize citations. The project is structured into several Python scripts and a Jupyter Notebook, each with a specific role in data processing and analysis.

# File Structure

data.py: Processes XML files, handles CSV file operations, and interacts with other scripts for data manipulation.
features.py: Contains functions for parsing and extracting features from textual data using libraries like BeautifulSoup and Scikit-learn.
read_data.py: Reads and processes data, mainly from CSV files, and performs data filtering and transformation.
relevant_refs.py: Stores a list of relevant references used across the project.
Classifier.ipynb: A Jupyter Notebook that likely includes the implementation of a classifier, along with data analysis and visualization.

# Usage

To use this project, run the scripts in the following order:

data.py to process the raw XML data.
features.py to extract features from the processed data.
read_data.py to read and further process the data from the CSV file.
Use the Classifier.ipynb notebook for building and evaluating the classifier.
