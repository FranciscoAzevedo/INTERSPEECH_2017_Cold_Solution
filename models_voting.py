# Imports
import numpy as np
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# CSV File Parsers
def parse_data(file_name):

    # Input: File Path
    # Outputs: Data Array

    with open(file_name, 'r') as f:        # Open File
        data = []
        for line in f:
            line = line.rstrip('\n')    # Remove new line characters
            line = line.split(',')        # Split line by ',' obtaining a list of strings as a result

            sample = [float(value) for value in line] # For each string in the list cast as a float
            
            data.append(sample)            # Add to Data array
        
    return np.array(data)                # Return data as Numpy Array


def parse_labels(file_name):

    with open(file_name, 'r') as f:            # Open File
        labels = [int(line.rstrip('\n')) for line in f]    # For each line in the file, 
                                                            # remove new line characters and cast the remaning string as an int
    return np.array(labels)


# Function to train the model
def train_model(data,labels):
    
    
    decision_tree = DecisionTreeClassifier()

    random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy')

    gaussian_nb = GaussianNB()

    grad_boost = GradientBoostingClassifier()

    estimators = [('GBoost', grad_boost), ('G_NB', gaussian_nb) , ('RF', random_forest), ('Trees', decision_tree)]

    # Instanciate the VotingClassifier using hard voting
    voter = VotingClassifier(estimators=estimators, voting='soft')
    voter.fit(data, labels)


    voter.fit(data,labels)
                            
    return voter

# Function to test the model on DEV SET
def test_model(data, labels, trained_model):
    predicted_labels = trained_model.predict(data)
    rounded_labels = np.clip(np.abs(np.round(predicted_labels)), 0, 1)
    f1 = metrics.precision_recall_fscore_support(labels,rounded_labels, labels=[0,1], average='macro')

    return f1, predicted_labels, rounded_labels


# Function to PREDICT LABELS OF TEST SET
def predict_model(data, trained_model):
    pred_labels = trained_model.predict(data)
    round_labels = np.clip(np.abs(np.round(pred_labels)), 0, 1)

    return round_labels


def main():
    
    # Set file tuples
    train_files = ('features_egemaps_train.csv', 'labels_train.txt')
    dev_files = ('features_egemaps_dev.csv', 'labels_dev.txt')
    test_files =('features_egemaps_test.csv')
    
    ## Load Data using Parsers
    
    # Training Data
    train_data = parse_data(train_files[0])
    train_labels = parse_labels(train_files[1])
    
    # Development Data
    dev_data = parse_data(dev_files[0])
    dev_labels = parse_labels(dev_files[1])

    # Test Data
    test_data = parse_data(test_files)
        
    # Train Model
    model = train_model(train_data, train_labels)
    
    # Get metrics for the development set
    f1, predicted_labels, rounded_labels = test_model(dev_data, dev_labels, model)

    round_labels = predict_model(test_data, model)
    
    # Save dev data into a file
    np.savetxt('labels_temp_dev.txt', rounded_labels, '%0.0f' , delimiter=',') 

    # Save test data into a file
    np.savetxt('labels_temp_test.txt', round_labels, '%0.0f' , delimiter=',') 

    print("UAR:", f1[1])
    
    return
    
if __name__ == "__main__":
    main()

