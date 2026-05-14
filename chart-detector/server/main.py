from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from detector import analyze_chart

app = FastAPI()


app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    url: str
    page: str
    site: str


@app.post("/analyze")
def analyze(req: ImageRequest):

    result = analyze_chart(
        req.url,
        req.page,
        req.site
    )

    return result
