from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
import numpy as np
import io
import requests
from openai import OpenAI
import json
import re

app = FastAPI()

# EasyOCR 한국어+영어
reader = easyocr.Reader(['en', 'ko'])

# OpenAI GPT-5 API
client = OpenAI(api_key="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")

# Google Custom Search API
API_KEY = 'AIzaSyDzkfnR5dZwUGFiTfNYpCwLGcey98D1zwI'
CX = '21dc25c04d7484156'

# OCR 텍스트 정규화
def normalize_ones(text: str) -> str:
    return ''.join(['1' if ch in 'Il|┃\\"' else ch for ch in text])

def normalize_text(text: str) -> str:
    return text.strip()

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        # OCR 실행
        results = reader.readtext(image_np)

        # 숫자만 있는 항목 제외 + 숫자와 문자 합치기 + 숫자와 문자 사이 띄어쓰기
        queries = []
        for res in results:
            text = normalize_text(normalize_ones(res[1]))
            # 숫자만 있으면 제외
            if not any(c.isalpha() for c in text):
                continue

            if len(queries) == 0:
                queries.append(text)
            else:
                # 이전 문자열에 숫자가 포함되어 있고 현재가 문자면 합치기
                if any(c.isdigit() for c in queries[-1]) and any(c.isalpha() for c in text):
                    combined = queries[-1] + text
                    # 숫자와 문자 사이 공백 넣기
                    combined = re.sub(r'(\d)([가-힣A-Za-z])', r'\1 \2', combined)
                    queries[-1] = combined
                else:
                    queries.append(text)

        final_output = {}

        for q in queries:
            try:
                # 검색어 최적화: 숫자와 이름 분리 후 OR 검색
                parts = re.findall(r'\d+|[^\d\s]+', q)
                optimized_query = ' '.join(parts)
                print(f"검색어 최적화: {optimized_query}")

                url = f'https://www.googleapis.com/customsearch/v1?q={optimized_query}&key={API_KEY}&cx={CX}'
                resp = requests.get(url, timeout=5)
                data = resp.json()

                # GPT-5 요청
                prompt = f"""
아래 데이터를 기반으로 약의 효능을 JSON 배열로 정리해 주세요.
- 각 항목은 {{
    "약이름": "...",
    "효능": ["..."],
    "효능_요약": "..."
  }} 구조로 하나만 생성
- 효능은 자연스러운 문장으로 작성
- 효능_요약은 가장 중요한 내용을 한 줄로 요약
- 각 약마다 효능과 요약은 하나씩만 생성하며, 중복되는 내용은 통합
- 오직 하나의 항목 데이터만 존재 해야함

데이터:
{data}
"""
                result = client.responses.create(
                    model="gpt-5-nano",
                    input=prompt
                )

                # JSON 파싱 시도
                try:
                    final_output[q] = json.loads(result.output_text)
                except json.JSONDecodeError:
                    final_output[q] = {"raw_text": result.output_text}

            except Exception as e:
                final_output[q] = {"error": str(e)}

        return JSONResponse(content={"queries": final_output})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "FastAPI EasyOCR + 검색 + GPT-5 서버 동작 중!"}
