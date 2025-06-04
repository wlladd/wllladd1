import os
import sys
import types

# Ensure src is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import requests

from data_fetcher import DataFetcher


class DummyResponse:
    status_code = 200

    def json(self):
        return {}

    def raise_for_status(self):
        pass


def test_default_header(monkeypatch):
    captured = {}

    def mock_get(url, params=None, headers=None):
        captured['headers'] = headers
        return DummyResponse()

    monkeypatch.setattr(requests, 'get', mock_get)
    fetcher = DataFetcher('https://example.com', 'KEY', 'SECRET')
    fetcher._signed_request('GET', '/api/test')
    assert captured['headers'] == {'X-MBXAPIKEY': 'KEY'}


def test_override_header(monkeypatch):
    captured = {}

    def mock_get(url, params=None, headers=None):
        captured['headers'] = headers
        return DummyResponse()

    monkeypatch.setattr(requests, 'get', mock_get)
    fetcher = DataFetcher('https://example.com', 'KEY', 'SECRET')
    fetcher._signed_request('GET', '/api/test', header_name='X-MBX-APIKEY')
    assert captured['headers'] == {'X-MBX-APIKEY': 'KEY'}
