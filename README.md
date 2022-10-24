# Fact-Retrieval-Augmentation-for-FinQA
Improving MultiHierTT model as a project for CS 678 Advanced NLP



## Structure
```
model/ # Contains model files
    fact_retriever.py # Retrieved facts
    span_selection.py # Selects the span
    question_classification.py # Classifies the questions' type to ease selecting operationi
    program_generation.py # Used to generate program off the extracted facts
data/ # Contains the data preprocessing files
utils.py # Utility functions for the model and data
train.py
setup.py # Contains constnat declarations
run.sh  # File to run the training
logger.py # Contains log methods
utils.py #contains utility functions
```

To run the code, configure train settings in train.py and then use:
```
sh run.sh
```
