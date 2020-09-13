# Machine Learning Engineer Offline Exercise

SalesLoft is looking to deploy a model to production that determines the seniority of a person based on their job title. This offline exercise will demonstrate some of your abilities related to this project.

## Requirements

- Python 3.5 or greater
- Installed dependencies (`pip install -r requirements.txt`)
- A [SalesLoft API](https://developers.salesloft.com/api.html#!/Topic/apikey) Key (the recruiter will provide this)
- Training data (`data/title_data_for_model_fitting.csv`)

## Getting Started

Copy (do not fork) this repository and follow the tasks listed below. Upon completion, please commit the code to a new GitHub repository or zip up the files and share with the recruiter.

## Your Task

In `exercise/model.py` you will find the `SeniorityModel` class. You can train this model using the data found in `data/title_date_for_model_fitting.csv`. Your job is to follow the below tasks which further enhance the capabilities of the model class.

1. Implement a `predict(job_titles)` class method that accepts an array of job title strings and returns the predicted seniorities.
1. Implement a `predict_salesloft_team()` class method that [loads all people](https://developers.salesloft.com/api.html#!/People/get_v2_people_json) in a team via the SalesLoft API and returns an array of tuples: `(id, seniority)`.
1. Implement a `save(filename)` class method which persists the information in the `SeniorityModel` to disk at the given location. The model should be saved in a language-agnostic format (i.e. don't use `pickle`) and contains all the information that is required to create a new instance of `SeniorityModel` from the saved data. Assume that someone could use this file to construct an equivalent model in another programming language, given documentation.
1. Implement a `load(filename)` class method that loads a saved model (see above) from the given location on disk and initializes the instance of `SeniorityMethod` using the information in the file so that it can be used for predictions without fitting.
1. Use a testing framework to assert that your save/load functions are working. Make sure that a set of predictions from the persisted model match that of the hydrated model.