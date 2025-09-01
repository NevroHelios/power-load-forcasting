from locust import HttpUser, task, between, tag
import itertools
import random

_user_counter = itertools.count(1)

SAMPLE_PAYLOAD = {
    "Date_Time": "01-01-2018 12:00",
    "Usage_kWh": 50.0,
    "Lagging_Current_Reactive_Power_kVarh": 10.0,
    "Leading_Current_Reactive_Power_kVarh": 5.0,
    "CO2_tCO2": 0.5,
    "Lagging_Current_Power_Factor": 0.9,
    "Leading_Current_Power_Factor": 0.85,
    "NSM": 43200.0
}

class PowerLoadUser(HttpUser):
    wait_time = between(0.2, 0.8)  

    def on_start(self):
        self.uid = next(_user_counter)

        if self.uid % 2 == 0:
            self.model_name = "xgboost"
        else:
            self.model_name = "lgbm"

        r = self.client.post("/set-model", json={"name": self.model_name})
        if r.status_code != 200:
            print(f"[on_start] model set failed uid={self.uid}: {r.status_code} {r.text}")

    @task(5)
    @tag("predict")
    def predict_load(self):
        r = self.client.post("/predict", json=SAMPLE_PAYLOAD)
        if r.status_code != 200:
            print(f"[predict] fail uid={self.uid}: {r.status_code} {r.text}")

    @task(1)
    @tag("switch")
    def switch_model(self):
        self.model_name = "lgbm" if self.model_name == "xgboost" else "xgboost"
        r = self.client.post("/set-model", json={"name": self.model_name})
        if r.status_code != 200:
            print(f"[switch] fail uid={self.uid}: {r.status_code} {r.text}")
            