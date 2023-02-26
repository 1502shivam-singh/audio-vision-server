from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

#loading the tokenizer and the model
def setLangTarget(fromLang, targetLang):
  modelName = "Helsinki-NLP/opus-mt-"+fromLang+"-"+targetLang
  tokenizer = AutoTokenizer.from_pretrained(modelName)
  model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
  return [tokenizer, model]

def translator(text, fromLang, targetLang):
  # function to translate english text to hindi
  langData = setLangTarget(fromLang, targetLang)
  input_ids = langData[0].encode(text, return_tensors="pt", padding=True)
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

# from fastapi import FastAPI, HTTPException
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import tensorflow as tf

# app = FastAPI()

# # Load the tokenizer and model once at startup
# model_name = "Helsinki-NLP/opus-mt-{}-{}"
# loaded_models = {}

# def get_model(from_lang, to_lang):
#     model_key = f"{from_lang}-{to_lang}"
#     if model_key not in loaded_models:
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_name.format(from_lang, to_lang))
#             model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name.format(from_lang, to_lang))
#             loaded_models[model_key] = (tokenizer, model)
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Failed to load model for {model_key}: {e}")
#     return loaded_models[model_key]

# @app.get('/translate')
# async def translate(from_lang: str, to_lang: str, text: str):
#     try:
#         tokenizer, model = get_model(from_lang, to_lang)
#         input_ids = tokenizer.encode([text], return_tensors="tf", padding=True)
#         outputs = model.generate(input_ids)
#         decoded_text = tokenizer.decode(outputs[0])
#         return {'translation': decoded_text}
#     except Exception as e:
#         print(e)
#         raise HTTPException(status_code=500, detail=f"Failed to translate: {e}")
