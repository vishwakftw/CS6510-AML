### Kaggle Challenge

My best submissions on the leaderboard used a Gradient Boosting Classifier and Random Forest Classifier, both Ensemble methods.
The Gradient Boosting Classifier took only a subset of features as input - 10 out of 19, whereas the input to my Random Forest Classifier had all features. The features for Gradient Boosting were hand picked from a 2D histogram. Mean reduction, label encoding and one hot vectorization was done using `sklearn`'s API.

Run the file: `CS15BTECH11043_Kaggle.py`
```bash
python3 CS15BTECH11043_Kaggle.py
```

After the end of the program, the submissions files will be there in the same directory named `sub_file_1.csv` and `sub_file_2.csv`.
