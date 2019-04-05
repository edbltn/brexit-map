import requests
from requests.auth import AuthBase

class NewsAPIException(Exception):

    def __init__(self, exception):
        self.exception = exception

    def get_exception(self):
        return self.exception

    def get_status(self):
        if self.exception["status"]:
            return self.exception["status"]

    def get_code(self):
        if self.exception["code"]:
            return self.exception["code"]

    def get_message(self):
        if self.exception["message"]:
            return self.exception["message"]

def get_auth_headers(api_key):
    return {
        'Content-Type': 'Application/JSON',
        'Authorization': api_key
    }

class GuardianApiClient(object):

    def __init__(self, api_key):
        self.api_key = api_key
        self.url = 'https://content.guardianapis.com/search'

    def get_everything(self, q=None, from_param=None, to=None, sort_by=None, page=None, page_size=None):

        payload = {}
        payload['api-key'] = self.api_key

        if q:
            payload['q'] = q
        if from_param:
            payload['from-date'] = from_param 
        if to:
            payload['to-date'] = to 
        if sort_by:
            payload['order-by'] = sort_by
        if page:
            payload['page'] = page
        if page_size:
            payload['page-size'] = page_size

        r = requests.get(self.url, timeout=30, params=payload)

        if r.status_code != requests.codes.ok:
            raise NewsAPIException(r.json())

        return r.json()
