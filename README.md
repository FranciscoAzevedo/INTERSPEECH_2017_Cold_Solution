# INTERSPEECH_2017_Cold_Solution
This repository includes a solution to the INTERSPEECH 2017 Cold Challenge. It includes GeMaps features already extracted for train, dev and test set as well as the code for 2 ensemble models used for the solution. It also includes a report with some conclusions and justification for this project. Overall we used 2 ensemble models with classifiers like decision trees, boosting, bagging and Naive Bayes.

NOTE 1: In the report Neural Networks and SVMS are discussed in very small detail (the train and eval curves are awfull) as we noticed that NN's were very hard to train in a way that generated good results, no matter the changes we made (we used L1,L2, Kernel and Weights regularization, different optimizers while grid searching the parameters and nothing changed a lot). On the other side SVM's were also yielding poor results and overall the best results on them were due to grid searches and not a solid understanding and reason behind those results. 

NOTE 2: the balance between healthy/cold person was altered by the teaching staff therefore we were not using the official INTERPSEECH corpus to train OR evaluate but a slightly differente one. 

Practical Notes: it ran on a virtual environment in python 3 using scikit-learn and other libraries like pandas, numpy and respective dependencies. Will upload the requirements file in the future
