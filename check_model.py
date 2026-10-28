import os

# 1. 強制避開底層套件衝突
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import google.generativeai as genai

# 2. 填入你 AQ. 開頭的 API Key
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

print("🔍 正在連線 Google 伺服器，查詢此金鑰專屬的可用模型清單...\n")

try:
    # 3. 呼叫官方 ListModels 函數
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            print(f"✅ 發現可用模型: {m.name}")

    if not available_models:
        print("⚠️ 你的金鑰目前沒有任何支援生成文字的模型權限。")
except Exception as e:
    print(f"❌ 連線失敗，錯誤詳情：{e}")