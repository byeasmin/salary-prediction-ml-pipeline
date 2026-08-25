import pytest
from fastapi.testclient import TestClient

from main import app
client = TestClient(app)
client.__enter__()

VALID_PAYLOAD = {
    "work_year": 2025,
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Data Scientist",
    "remote_ratio": 100,
    "company_size": "M",
    "company_location": "US",
    "employee_residence": "US",
}

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
def test_predict_valid_payload():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_salary_usd" in body
    assert body["predicted_salary_usd"] > 0
    assert body["predicted_salary_range_usd"]["low"] < body["predicted_salary_usd"]
    assert body["predicted_salary_range_usd"]["high"] > body["predicted_salary_usd"]
def test_predict_entry_level_lower_than_executive():
    entry_payload = {**VALID_PAYLOAD, "experience_level": "EN"}
    exec_payload = {**VALID_PAYLOAD, "experience_level": "EX"}

    entry_resp = client.post("/predict", json=entry_payload).json()
    exec_resp = client.post("/predict", json=exec_payload).json()

    assert entry_resp["predicted_salary_usd"] < exec_resp["predicted_salary_usd"]
def test_predict_invalid_experience_level():
    bad_payload = {**VALID_PAYLOAD, "experience_level": "SENIOR"}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422  


def test_predict_invalid_remote_ratio():
    bad_payload = {**VALID_PAYLOAD, "remote_ratio": 150}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422  


def test_predict_invalid_company_location():
    bad_payload = {**VALID_PAYLOAD, "company_location": "ZZ"}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_missing_required_field():
    bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "job_title"}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_unknown_job_title_falls_back_to_other():
    payload = {**VALID_PAYLOAD, "job_title": "Some Brand New Title"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_same_country_flag_affects_prediction():
    same_country = {**VALID_PAYLOAD, "employee_residence": "US", "company_location": "US"}
    diff_country = {**VALID_PAYLOAD, "employee_residence": "IN", "company_location": "US"}

    r1 = client.post("/predict", json=same_country).json()
    r2 = client.post("/predict", json=diff_country).json()

    assert r1["predicted_salary_usd"] > 0
    assert r2["predicted_salary_usd"] > 0
