import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fuzzywuzzy import fuzz
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from Models.model import classifier
from Models.x_calculator import calculate_x

MANUAL_ENCODING = {
    "information technology and services": 0,
    "military": 1,
    "accounting": 2,
    "retail": 3,
    "computer software": 4,
    "telecommunications": 5,
    "defense & space": 6,
    "financial services": 7,
    "management consulting": 8,
    "banking": 9
}


class DeltaModel(BaseModel):
    industry: str
    Emissions: float
    disaster_risk: float
    importance: float


dataset = pd.read_csv("../Models/companies_final.csv")
dataset.drop("Unnamed: 0", axis=1, inplace=True)

dataset = dataset.iloc[:2000]  # because only 2000 entries are used in the training data

app = FastAPI()


def convert_encoding(industry: str) -> int:
    return MANUAL_ENCODING.get(industry, -1)  # Returns -1 for unknown industries


# Numpy arrays have issues when they're converted directly into JSON
# Hence, we use this workaround to solve that
def round_all(data: dict) -> dict:
    rounded_dict = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Convert to a Python list
            rounded_dict[key] = np.round(value, 5).tolist()
        elif isinstance(value, (np.int32, np.int64, np.float32, np.float64)):
            # Convert to Python native int or float
            rounded_dict[key] = np.round(value, 5).item()
        else:
            rounded_dict[key] = round(value, 5)
    return rounded_dict


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/ping")
async def pong():
    return {"message": "pong"}


@app.post("/predict")
async def handle_model(data: DeltaModel):
    try:
        data_dict = data.model_dump()
        data_dict["industry"] = convert_encoding(data_dict["industry"])
        prediction = classifier.predict(data_dict)
        return round_all(prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calc_x")
async def calc_x(data: DeltaModel):
    try:
        data_dict = data.model_dump()
        data_dict["industry"] = convert_encoding(data_dict["industry"])
        x_value = calculate_x(data_dict)
        return {"x_value": x_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def get_company_results(query: str, limit: int = 10):
    names = dataset["name"].tolist()
    print(names)
    names = sorted(names, key=lambda x: fuzz.ratio(x.lower(), query.lower()), reverse=True)[:limit]
    results = []
    for name in names:
        results.append(dataset[dataset["name"] == name].iloc[0].to_dict())
    return JSONResponse(results)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
