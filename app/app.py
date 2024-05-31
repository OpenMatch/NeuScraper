# Copyright (c) 2023 OpenMatch
# Author: Zhipeng Xu
# All rights reserved.

from fastapi import FastAPI
from pydantic import BaseModel
from builder import build
import requests
import torch
from fastapi import FastAPI, HTTPException
from extractor import inference, save_predictions, get_text_spans_from_nodes
from extractor import ContentExtractionDeepModel
from arguments import create_parser

app = FastAPI()

class InputData(BaseModel):
    url: str


def init_model():
    parser = create_parser()
    args, _ = parser.parse_known_args()
    args.model_path = "/path/to/your/model"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ContentExtractionDeepModel(args)
    return model, args

model, args = init_model()

@app.post("/predict/")
async def predict(input_data: InputData):
    response = requests.get(input_data.url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Error fetching URL")

    html_content = response.content
    text_nodes_df, data = build(input_data.url, html_content)
    
    pred_nodes = inference(args, model, data)
    pred_nodes_df = save_predictions(pred_nodes)
    
    pred_df = get_text_spans_from_nodes(text_nodes_df, pred_nodes_df).dropna().sort_values(['TextNodeId'], ascending=[False])
    pred_df = pred_df.groupby(['Url', 'Task'], as_index=False).agg({'Text': ''.join})
    
    return {"Text": pred_df['Text'][0]}




