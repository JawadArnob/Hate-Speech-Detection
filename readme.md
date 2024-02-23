# In case something went wrong:
For any reason if the code did run this is a shared link for the full code in google colab:
https://colab.research.google.com/drive/12QgBGOK1Op-gWDrfsQsH_jtkdDsem0Y0?usp=sharing




### There are two files which are as follows: 
A-The file named as Analysis_CMT_316_assignment_2.ipynb is for data analysis only.
B-The file named as Full_CMT_316_assignment_2.ipynb is for the whole project including data analysis only.

The following are the assumptions taken into consideration for the project execution to be successful.

1. The project mainly relies on the NLTK library for the natural language processing. The assumption is that the 'NLTK' library is pre-installed, and any other sub-requirements for NLTK (eg: stopwords, punkt, etc) are then installed using the command mentioned in the colab cell.

2. Ther are other important library must be installed which are: 
 -tensorflow -------> to install: pip install tensorflow 
 
 -textblob ---------> to install: pip install textblob
 
 -contractions -----> to install: pip install contractions
 
 -wordcloud --------> to install: pip install wordcloud 
 
 -spacy ------------> to install: pip install spacy
 
 -seaborn ----------> to install: pip install seaborn
 
 -sklearn ----------> to install: pip install scikit-learn
 
 -pandas -----------> to install: pip install pandas
 
However the installation codes are commeneted in the first cell

3. IF the project runs on google, the path for the dataset would be "/content/drive/MyDrive/dataset/". This means, the folder "dataset" will be in the root folder in the google drive. However, the instructions about this are as commenets in the second cell

4. IF the project runs on local machine, the path for the dataset would be "/dataset/". This means, this folder "dataset" and the code should be in same folder. However, the instructions about this are as commenets in the second cell


5. The project would contain 2 additional files apart from CMT_316_assignment_2.ipynb, which are "text_processor.py" and "emoji_data.py". Both files contain functions which are prerequisites for the execution, and would be saved in the same root folder as the dataset.

# Execution

Please note, the submitted ipynb should be executed on local machine. However, there is a comments in the second cell in the code for run it in google colab which are as follows:
In the first cell there are these line for running the code in google colab (commnted by default), if the code will be run in locally this code must be commented
from google.colab import drive
drive.mount('/content/drive') 
path = "/content/drive/MyDrive/dataset/" #  main folder if the code ran in google colab
path = "/dataset/" # main folder if the code ran in local machine

Addtionally, there is part in the full code called "keras-tuner", it is recommended not to run it as it takes too much time and it was used for hyperparameters tuning 

Action - "Run all cells"





## Group Members

Student id: 21118923, Aloraini, Osama, [alorainiom@cardiff.ac.uk](mailto:alorainiom@cardiff.ac.uk)
Student id: 22107200, Tok, Emin Safa, [toke@cardiff.ac.uk](mailto:toke@cardiff.ac.uk)
Student id: 22074262, Sawant, Uday, [sawantu@cardiff.ac.uk](mailto:sawantu@cardiff.ac.uk)
Student id: 22099173, Md Akib Al Jawad Arnob, [ArnobM@cardiff.ac.uk](mailto:ArnobM@cardiff.ac.uk)
Student id: 22071956, Kadam, Swapnil, [KadamS1@cardiff.ac.uk](mailto:KadamS1@cardiff.ac.uk)
Student id: 22081164, Huajian, Li, [Lih92@cardiff.ac.uk](mailto:Lih92@cardiff.ac.uk)
Student id: 22055417, Deshpande, Kaustubh, [deshpandek@cardiff.ac.uk](mailto:deshpandek@cardiff.ac.uk)
