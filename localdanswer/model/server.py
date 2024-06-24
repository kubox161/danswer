from typing import Optional, Dict, Any
import uvicorn

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import tensorflow as tf



GENERATIVE_AI_MODEL_REPO = "TheBloke/LLaMA-Pro-8B-GGUF"
GENERATIVE_AI_MODEL_FILE = "llama-pro-8b.Q3_K_M.gguf"

model_path = hf_hub_download(
    repo_id=GENERATIVE_AI_MODEL_REPO,
    filename=GENERATIVE_AI_MODEL_FILE
)

llama2_model = Llama(
    model_path=model_path,
    n_gpu_layers=64,
    n_ctx=2000
)


print(llama2_model(prompt="Hello ", max_tokens=1))


app = FastAPI()



class TextInput(BaseModel):
    inputs: str
    parameters: Optional[Dict[str, Any]]


@app.get("/")
def status_gpu_check() -> Dict[str, str]:
    gpu_msg = "Available" if tf.test.is_gpu_available() else "Unavailable"
    return {
        "status": "I am ALIVE!",
        "gpu": gpu_msg
    }


@app.post("/generate/")
async def generate_text(data: TextInput) -> Dict[str, str]:
    try:
        params = data.parameters or {}
        print("DATAAA")
        print(data)
        response = llama2_model(prompt=data.inputs, **params)
        model_out = response['choices'][0]['text']
        print({"generated_text": model_out})
        return {"generated_text": model_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
