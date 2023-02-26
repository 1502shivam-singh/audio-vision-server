from fastapi import FastAPI
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

app = FastAPI()

#loading the tokenizer and the model
def setLangTarget(fromLang, targetLang):
    modelName = "Helsinki-NLP/opus-mt-"+fromLang+"-"+targetLang
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(modelName)
    return [tokenizer, model]

def translator(text, fromLang, targetLang):
    # function to translate text from a source language to a target language
    langData = setLangTarget(fromLang, targetLang)
    input_ids = langData[0].encode(text, return_tensors="tf", padding=True)
    outputs = langData[1].generate(input_ids)
    decoded_text = langData[0].decode(outputs[0], skip_special_tokens=True)
    
    return decoded_text

@app.get('/predict/{input_data}')
async def predict(input_data: str):
    # Process the input data
    params = input_data.split('&')
    output_data = translator(params[2], params[0], params[1])

    # Return the output data
    return {'output_data': output_data}