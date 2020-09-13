import pytest
import os

from exercise.model import SeniorityModel
import pandas as pd

def test_save_load():
    mdl = SeniorityModel()
    df = pd.read_csv('data/title_data_for_model_fitting.csv')
    mdl.fit(*df.values.T)

    mdl.save('test.json')
    hydrated_mdl = SeniorityModel.load('test.json')

    mdl_preds = mdl.predict(df.job_title.values)
    hydrated_mdl_preds = hydrated_mdl.predict(df.job_title.values)

    assert all(mdl_preds == hydrated_mdl_preds) # checks elementwise
    os.remove('test.json') # remove file
