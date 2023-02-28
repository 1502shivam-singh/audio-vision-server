from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

#
#   hi -> en
#   en -> hi
#   en -> de
#   en -> ja
#   en -> zh
#

# Set up CORS
origins = [
    "http://127.0.0.1:5500",
    "https://audiovision-bridge.netlify.app/"
    # Add any other allowed origins here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#loading the tokenizer and the model
def setLangTarget(fromLang, targetLang):
    modelName = "Helsinki-NLP/opus-mt-"+fromLang+"-"+targetLang
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
    return [tokenizer, model]

langPairs = [["hi","en"], ["en","hi"], ["en","de"]]

modelMap = {}

for langSet in langPairs:
    modelMap[tuple(langSet)] = setLangTarget(langSet[0], langSet[1])

def translator(text, fromLang, targetLang):
    # function to translate text from a source language to a target language
    langData = modelMap[(fromLang, targetLang)]
    input_ids = langData[0].encode(text, return_tensors="pt", padding=True)
    outputs = langData[1].generate(input_ids)
    decoded_text = langData[0].decode(outputs[0], skip_special_tokens=True)
    
    return decoded_text

@app.get('/predict/{input_data}')
async def predict(input_data: str):
    try:
        # Process the input data
        params = input_data.split('&')
        output_data = translator(params[2], params[0], params[1])

        # Return the output data
        return {'output_data': output_data}
    except Exception as e:
        # Return an error message if an exception is raised
        return {'error': str(e)}
    