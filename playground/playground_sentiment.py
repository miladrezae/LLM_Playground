from transformers import pipeline
import torch
from libs.helper import timer_func



sentiment_text_1 = 'It was weird at first, but It grows on you'
sentiment_text_2 = "I guess it works, but can't rely on it in time of need. don't buy if you're sensitive"
input_text = [sentiment_text_1,sentiment_text_2]

sentiment_pipeline_gpu = pipeline(task='sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device="cuda") #0
sentiment_pipeline_cpu = pipeline(task='sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device="cpu") #-1

@timer_func
def inference(pipeline, input):
    print(pipeline(input))


inference(sentiment_pipeline_gpu,input_text)
inference(sentiment_pipeline_cpu,input_text)
