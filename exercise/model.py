import pandas as pd
import numpy as np
import re
import json
import requests

from dotenv import load_dotenv
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

load_dotenv()

API_KEY = os.getenv("API_KEY")

class ModelNotFitError(Exception):
    """Raised when a model has not been fit prior to prediction"""
    def __init__(
        self,
        message="Call the fit method on training data prior "
                "to making predictions."
    ):
        self.message = message
        super().__init__(self.message)

class NoAPIKeyError(Exception):
    """Raised when no API KEY is found in evironment"""
    pass


def clean_transform_title(job_title):
    """Clean and transform job title. Remove punctuations,
    special characters, multiple spaces etc.
    """
    if not isinstance(job_title, str):
        return ''
    new_job_title = job_title.lower()
    special_characters = re.compile('[^ a-zA-Z]')
    new_job_title = re.sub(special_characters, ' ', new_job_title)
    extra_spaces = re.compile(r'\s+')
    new_job_title = re.sub(extra_spaces, ' ', new_job_title)

    return new_job_title


class SeniorityModel:
    """Job seniority model class. Contains attributes to fit, predict,
    save and load the job seniority model.
    """
    def __init__(self):
        self.vectorizer = None
        self.model = None

    def _check_for_access(self):
        # Checks if there is an API_KEY
        if not API_KEY:
            raise NoAPIKeyError(
                "No API_KEY found in evironment"
            )

        # Checks if the API_KEY provides access to /v2/people.json
        r = requests.get(
            url='https://api.salesloft.com/v2/people.json',
            json={'page': 1, 'per_page': 1},
            headers={'Authorization': f'Bearer {API_KEY}'}
        )
        r.raise_for_status()

    def _check_for_array(self, variable):
        if not isinstance(variable, (list, tuple, np.ndarray)):
            raise TypeError(
                "variable should be of type list or numpy array."
            )
        return

    def _data_check(self, job_titles, job_seniorities):
        self._check_for_array(job_titles)
        self._check_for_array(job_seniorities)

        if len(job_titles) != len(job_seniorities):
            raise IndexError(
                "job_titles and job_seniorities must be of the same length."
            )

        return

    def _check_model(self):
        if not self.vectorizer or not self.model:
            raise ModelNotFitError()
        return

    def fit(self, job_titles, job_seniorities):
        """Fits the model to predict job seniority from job titles.
        Note that job_titles and job_seniorities must be of the same
        length.

        Parameters
        ----------
        job_titles: numpy array or list of strings representing
                    job titles
        job_seniorities: numpy array or list of strings representing
                         job seniorities
        """
        self._data_check(job_titles, job_seniorities)

        cleaned_job_titles = np.array([
            clean_transform_title(job_title)
            for job_title in job_titles
        ])

        self.vectorizer = CountVectorizer(
            ngram_range=(1,2),
            stop_words='english'
        )
        vectorized_data = self.vectorizer.fit_transform(cleaned_job_titles)
        self.model = LinearSVC()
        self.model.fit(vectorized_data, job_seniorities)

        return

    def predict(self, job_titles):
        """Assigns predicted job seniorities from job titles.

        Parameters
        ----------
        job_titles: numpy array or list of strings representing job
                    titles
        """
        self._check_for_array(job_titles)
        self._check_model()

        cleaned_job_titles = np.array([
            clean_transform_title(job_title)
            for job_title in job_titles
        ])

        vectorized_data = self.vectorizer.transform(cleaned_job_titles)
        return self.model.predict(vectorized_data)

    def predict_salesloft_team(self):
        """Assigns predicted job seniorities from the job titles
        available from the
        """
        self._check_api()
        self._check_model()

        page = 1
        ids = []
        titles = []

        while page:
            r = requests.get(
                url='https://api.salesloft.com/v2/people.json',
                json={'page': page},
                headers={'Authorization': f'Bearer {API_KEY}'}
            )

            r.raise_for_status()

            data = r.json()

            page = data['metadata']['paging']['next_page']

            ids.extend([d['id'] for d in data['data']])
            titles.extend(d['title'] for d in data['data'])

        seniorities = self.predict(titles)

        returned_data = np.empty(len(ids), dtype=object)
        returned_data[:] = list(zip(ids, seniorities))

        return returned_data

    def save(self, filename):
        """Saves relevant model data to json file.

        Parameters
        ----------
        filename: str representing file path/name
        """
        self._check_vectorizer()
        self._check_model()

        data = {}
        data['vectorizer'] = dict(
            (k, v)
            for k, v in self.vectorizer.__dict__.items()
            if k in [
                'stop_words', 'ngram_range',
                'vocabulary_', 'fixed_vocabulary_'
            ]
        )

        data['vectorizer']['stop_words_'] = list(self.vectorizer.stop_words_)

        data['model'] = dict(
            (k, v.tolist())
            for k, v in self.model.__dict__.items()
            if k in [
                'coef_', 'intercept_', 'classes_'
            ]
        )
        data['model']['n_iter_'] = int(self.model.n_iter_)

        with open(filename, 'w') as f:
            json.dump(data, f)

        return

    @classmethod
    def load(cls, filename):
        """Class method to initialize a model from a JSON file,
        used to hydrate from previously fit model.

        Parameters
        ----------
        filename: str representing file path/name
        """
        mdl = cls()

        with open(filename, 'r') as f:
            data = json.load(f)

        mdl.vectorizer = CountVectorizer()
        for attr, val in data['vectorizer'].items():
            setattr(mdl.vectorizer, attr, val)

        mdl.model = LinearSVC()
        for attr, val in data['model'].items():
            if isinstance(val, list):
                setattr(mdl.model, attr, np.array(val))
            else:
                setattr(mdl.model, attr, val)

        return mdl
