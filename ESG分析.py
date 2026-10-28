import os
import json
import re
import time
import sys
import warnings
import pymupdf as fitz  # 使用 pymupdf 避免 'fitz API is deprecated' 警告
import pandas as pd
import google.generativeai as genai
import gc
from PIL import Image
import io
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# ==========================================
# 🌟 0. 隱藏 Future / Deprecation 警告與環境變數設定
# ==========================================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SKIP_FIRST_N_PAGES = 5

# ==========================================
# 🔑 1. API 金鑰池 (Key Pool) 與模型初始化
# 注意：請將您的 Gemini API Keys 填入下方陣列中或設定環境變數
# ==========================================
API_KEYS_POOL = [
    " 輸入你申請的google api key "
    "第二條",
    "第三條",
]

# 若金鑰池為空，嘗試從系統環境變數讀取
if not API_KEYS_POOL:
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        API_KEYS_POOL.append(env_key)
    else:
        print("❌ 錯誤：未設定 API Key，請在 API_KEYS_POOL 中填入您的 Key 或設定 GEMINI_API_KEY 環境變數。")
        sys.exit(1)

current_key_index = 0
genai.configure(api_key=API_KEYS_POOL[current_key_index])

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# 全使用極速且省流量的文字模型 gemini-3.5-flash-lite
model = genai.GenerativeModel(
    'gemini-3.5-flash-lite',
    safety_settings=safety_settings
)

# 初始化 RapidOCR (純 ONNX 引擎，不需安裝任何 C++ 補丁)
print("🔍 正在初始化 RapidOCR 輕量化辨識引擎...")
rapid_ocr = RapidOCR()

LOCATION_BLACK_LIST = ["台北", "台中", "高雄", "台南", "分公司", "辦事處", "總公司", "工廠"]
python_excel_cleaner = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

api_call_count = 0


# ==========================================
# 🛠 2. 清洗與過濾工具
# ==========================================
def is_noisy_text(text):
    if len(text) < 15: return True
    if re.search(r'\.{4,}', text) or "目錄" in text[:10]: return True
    if bool(re.match(r'^[\d\s\W_]+$', text)): return True
    if sum(1 for loc in LOCATION_BLACK_LIST if loc in text) >= 1 and len(text) < 35: return True
    return False


# ==========================================
# ✂ 3. 精確斷句模組 (僅保留 。！？)
# ==========================================
def split_into_short_units(text, max_sentences=2):
    text = text.replace('\n', '')
    parts = re.split(r'([。！？])', text)
    sentences = []
    current_sentence = ""

    for part in parts:
        current_sentence += part
        if part in ['。', '！', '？']:
            if len(current_sentence.strip()) > 5:
                sentences.append(current_sentence.strip())
            current_sentence = ""

    if len(current_sentence.strip()) > 5:
        sentences.append(current_sentence.strip())

    short_units = []
    for i in range(0, len(sentences), max_sentences):
        unit = "".join(sentences[i:i + max_sentences])
        if len(unit.strip()) > 10 and not is_noisy_text(unit):
            short_units.append(unit.strip())

    return short_units


# ==========================================
# 🚀 4. Gemini 3.5 Lite 批次文字語調分析
# ==========================================
def analyze_esg_tone_batch(batch_texts, retries=5):
    global api_call_count, current_key_index, model
    if not batch_texts:
        return []

    numbered_texts = "\n".join([f"[{i}] {t}" for i, t in enumerate(batch_texts)])
    prompt = f"""
    你是一位專業的台灣財報與 ESG 語調分析專家。
    請閱讀以下帶有編號的永續報告書短句，判斷其：
    1. ESG 主分類：E、S、G 或 None (非ESG相關)。
    2. 語調標籤：Positive、Negative、Neutral。

    分析準則：
    - 「無發生意外、0次裁罰」應視為 Positive。
    - 「裁員、碳排增加、客訴」應視為 Negative。
    - 即使內容為非 ESG 相關 (分類為 None)，也必須依照句子本身的語氣，判斷其為 Positive、Negative 或 Neutral。

    請務必以合法的 JSON 陣列格式回傳，格式如下：
    [
      {{"id": 0, "category": "E", "tone": "Positive"}},
      {{"id": 1, "category": "None", "tone": "Neutral"}}
    ]

    以下是欲分析文字：
    {numbered_texts}
    """

    for attempt in range(retries):
        try:
            api_call_count += 1
            response = model.generate_content(prompt)

            # 檢查是否遭到 Prompt 阻擋
            if hasattr(response, "prompt_feedback") and response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"⚠️ Prompt 被系統阻擋: {response.prompt_feedback.block_reason}")
                return [{"id": i, "category": "None", "tone": "Neutral"} for i in range(len(batch_texts))]

            result_str = response.text.strip()

            if result_str.startswith("```"):
                result_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", result_str, flags=re.MULTILINE)

            json_results = json.loads(result_str)
            return json_results

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "resource_exhausted" in error_str.lower():
                if current_key_index < len(API_KEYS_POOL) - 1:
                    current_key_index += 1
                    print(f"\n   ⚠️ 第 {current_key_index} 把金鑰觸發限額！切換至第 {current_key_index + 1} 把金鑰...")
                    genai.configure(api_key=API_KEYS_POOL[current_key_index])
                    model = genai.GenerativeModel('gemini-3.5-flash-lite', safety_settings=safety_settings)
                    time.sleep(2)
                else:
                    current_key_index = 0
                    print(f"\n   ⏳ 所有金鑰皆觸發 RPM 限額，啟動冷卻程序... 靜置 60 秒後重試...")
                    genai.configure(api_key=API_KEYS_POOL[current_key_index])
                    model = genai.GenerativeModel('gemini-3.5-flash-lite', safety_settings=safety_settings)
                    time.sleep(60)
            else:
                print(f"⚠️ 解析錯誤 (重試中): {e}")
                time.sleep(3)

    return [{"id": i, "category": "STOP", "tone": "QUOTA_EXHAUSTED"} for i in range(len(batch_texts))]


# ==========================================
# 🔍 5. 混合式 PDF 解析模組 (PyMuPDF 文字 + RapidOCR 免費本地救援)
# ==========================================
def extract_and_split_pdf(pdf_path):
    extracted_units = []
    doc = fitz.open(pdf_path)

    try:
        total_pages = len(doc)
        print(f"   📄 PDF 總頁數：{total_pages} 頁，開始處理...")

        for page_num in range(total_pages):
            if page_num < SKIP_FIRST_N_PAGES:
                continue
            page = doc[page_num]

            # 1️⃣ 第一階段：PyMuPDF 直接文字提取
            text = page.get_text("text").strip()
            raw_paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 10]

            page_units = []
            for para in raw_paras:
                units = split_into_short_units(para, max_sentences=2)
                page_units.extend(units)

            # 2️⃣ 第二階段：若該頁無文字，使用 RapidOCR 在本地進行救援
            if not page_units:
                print(f"      ...第 {page_num + 1} 頁無文字，啟動 RapidOCR 本地救援...")
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")

                # RapidOCR 本地辨識
                result, _ = rapid_ocr(img_bytes)

                if result:
                    ocr_text = "".join([line[1] for line in result])
                    page_units = split_into_short_units(ocr_text, max_sentences=2)

                del pix
                gc.collect()

            if page_units:
                extracted_units.extend(page_units)

    finally:
        doc.close()
        gc.collect()

    return extracted_units


# ==========================================
# 📊 6. 主程式：檔名檢查 ＋ 續傳總表
# ==========================================
if __name__ == "__main__":
    folder_path = r"G:\我的雲端硬碟\參考論文\永續報告書"
    company_excel_path = "company_list.xlsx"
    master_excel_name = "ESG_量化統計結果_總表.xlsx"
    time_log_file = "total_time_log.json"  # ⏱️ 記錄歷史累積用時的 json 檔

    if not os.path.exists(folder_path):
        print(f"❌ 找不到路徑：{folder_path}，請確認「Google 雲端硬碟電腦版」已啟動且路徑正確！")
    else:
        # 📂 1. 偵測產業資料夾
        available_industries = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

        print("=" * 60)
        print("📂 目前雲端硬碟中找到的【產業資料夾列表】：")
        for idx, ind in enumerate(available_industries, 1):
            print(f"   {idx}. {ind}")
        print("=" * 60)

        user_input = input(
            "\n👉 請輸入今天想分析的【產業名稱】或【列表編號】\n"
            "   （如需多個請用逗號隔開，例如: 1,3 或 半導體,金融業；若直接按 Enter 鍵代表分析『全部』產業）：\n> "
        ).strip()

        TARGET_INDUSTRIES = []

        if user_input:
            raw_items = [i.strip() for i in re.split(r'[,，]', user_input) if i.strip()]
            for item in raw_items:
                if item.isdigit():
                    idx = int(item) - 1
                    if 0 <= idx < len(available_industries):
                        selected_ind = available_industries[idx]
                        if selected_ind not in TARGET_INDUSTRIES:
                            TARGET_INDUSTRIES.append(selected_ind)
                    else:
                        print(f"⚠️ 無效的編號：{item} (超出範圍，已忽視)")
                else:
                    if item in available_industries:
                        if item not in TARGET_INDUSTRIES:
                            TARGET_INDUSTRIES.append(item)
                    else:
                        print(f"⚠️ 找不到名為 '{item}' 的產業資料夾 (已忽視)")

        # 🔍 2. 穿透掃描檔案
        pdf_file_info = []
        if TARGET_INDUSTRIES:
            print(f"\n🎯 已經選擇以下產業目標: {TARGET_INDUSTRIES}")
            for industry in TARGET_INDUSTRIES:
                ind_path = os.path.join(folder_path, industry)
                if os.path.exists(ind_path):
                    for root, dirs, files in os.walk(ind_path):
                        for file in files:
                            if file.lower().endswith(".pdf"):
                                full_path = os.path.join(root, file)
                                pdf_file_info.append((full_path, file))
        else:
            if user_input:
                print("\n⚠️ 輸入的編號或名稱皆不符合，未找到任何匹配產業！")
                sys.exit()
            else:
                print("\n🔍 未輸入產業名稱，啟動【全產業】穿透掃描...")
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(".pdf"):
                            full_path = os.path.join(root, file)
                            pdf_file_info.append((full_path, file))

        # 🛡️ 3. 檔名格式健康檢查
        if pdf_file_info:
            print(f"\n🔍 正在對 {len(pdf_file_info)} 個 PDF 檔案進行檔名格式健康檢查...")
            invalid_files = []
            filename_pattern = re.compile(r'^\d+_\d{4}\.pdf$', re.IGNORECASE)

            for full_path, file_name in pdf_file_info:
                if not filename_pattern.match(file_name):
                    invalid_files.append((file_name, full_path))

            if invalid_files:
                print("\n" + "❌" * 30)
                print(f"🚨 發現 {len(invalid_files)} 個檔案不符合 '股票代碼_年份.pdf' 格式！")
                print("請先更正以下檔案的名稱後，再重新執行程式：")
                print("-" * 60)
                for f_name, f_path in invalid_files:
                    print(f" ❌ 不合規檔名: {f_name}")
                    print(f"    完整檔案路徑: {f_path}\n")
                print("❌" * 30)
                print("\n🛑 程式已自動停止，未扣除任何 API 流量。修正後請重新執行！")
                sys.exit()
            else:
                print("✅ 檔名格式健康檢查通過！")
        else:
            print("⚠️ 找不到符合條件的 PDF 檔案。")
            sys.exit()

        start_time = time.time()

        # 📖 4. 讀取公司清單建立字典
        company_dict = {}
        if os.path.exists(company_excel_path):
            try:
                df_comp = pd.read_excel(company_excel_path)
                df_comp['股票代碼'] = df_comp['股票代碼'].astype(str).str.strip()
                for _, row in df_comp.iterrows():
                    company_dict[row['股票代碼']] = row['公司簡稱']
                print(f"✅ 成功載入上市櫃公司清單，共 {len(company_dict)} 筆資料。")
            except Exception as e:
                print(f"⚠ 讀取公司清單發生錯誤：{e}")
        else:
            print(f"⚠️ 找不到公司清單：{company_excel_path} (將以空白代替公司簡稱)")

        # 🔄 5. 讀取或創建【總表】
        columns_order = [
            "股票代碼", "年度", "公司簡稱",
            "E_positive", "E_negative", "E_neutral",
            "S_positive", "S_negative", "S_neutral",
            "G_positive", "G_negative", "G_neutral",
            "non_positive", "non_negative", "non_neutral",
            "總句數"
        ]

        processed_keys = set()
        if os.path.exists(master_excel_name):
            try:
                existing_df = pd.read_excel(master_excel_name)
                for _, row in existing_df.iterrows():
                    key = f"{str(row['股票代碼']).strip()}_{str(row['年度']).strip()}"
                    processed_keys.add(key)
                print(f"📦 偵測到歷史總表 '{master_excel_name}'，包含 {len(processed_keys)} 筆已完成資料。啟動自動跳過機制！")
            except Exception as e:
                print(f"⚠️ 讀取歷史總表發生錯誤，將建立新總表：{e}")
                existing_df = pd.DataFrame(columns=columns_order)
        else:
            existing_df = pd.DataFrame(columns=columns_order)
            print(f"📄 未找到歷史總表，將建立全新的總表：'{master_excel_name}'")

        quota_exhausted = False
        new_analyzed_data = []

        # 🚀 6. 開始 AI 統計分析
        print(f"\n📂 啟動 AI 統計分析，正在比對跳過重複檔案...")

        for full_path, file_name in pdf_file_info:
            if quota_exhausted:
                break

            match = re.match(r'^(\d+)_(\d{4})', file_name)
            stock_code = match.group(1)
            year = match.group(2)
            file_key = f"{stock_code}_{year}"

            if file_key in processed_keys:
                print(f"   ⏭️ [跳過] {file_name} (代碼:{stock_code} | 年度:{year}) 數據已存在總表中！")
                continue

            company_name = company_dict.get(stock_code, "未匹配")
            print(f"\n▶ 正在分析新檔案: {file_name} (代碼:{stock_code} | 年度:{year} | 公司:{company_name})")

            stats = {
                "E_positive": 0, "E_negative": 0, "E_neutral": 0,
                "S_positive": 0, "S_negative": 0, "S_neutral": 0,
                "G_positive": 0, "G_negative": 0, "G_neutral": 0,
                "non_positive": 0, "non_negative": 0, "non_neutral": 0,
                "總句數": 0
            }

            category_map = {"E": "E", "S": "S", "G": "G", "None": "non"}
            tone_map = {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}

            # 1️⃣ 提取全檔文字 (含 RapidOCR 本地救援)
            units = extract_and_split_pdf(full_path)

            # 2️⃣ 分批送給 Gemini 3.5 Lite 分析
            if units:
                print(f"   💬 全檔共整理出 {len(units)} 個段落，開始分批進行 Gemini 語調分析...")
                BATCH_SIZE = 10
                for i in range(0, len(units), BATCH_SIZE):
                    batch_texts = units[i:i + BATCH_SIZE]
                    ai_results = analyze_esg_tone_batch(batch_texts)

                    if ai_results and isinstance(ai_results, list) and ai_results[0].get("category") == "STOP":
                        print("\n🛑 啟動緊急斷點存檔程序... 將即時保存已處理的資料至總表。")
                        quota_exhausted = True
                        break

                    for res in ai_results:
                        try:
                            c = res.get("category", "")
                            t = res.get("tone", "")
                            if c in category_map and t in tone_map:
                                stat_key = f"{category_map[c]}_{tone_map[t]}"
                                if stat_key in stats:
                                    stats[stat_key] += 1
                                    stats["總句數"] += 1
                        except Exception:
                            pass

                    print(f"      ...分析進度：已完成 {min(i + BATCH_SIZE, len(units))} / {len(units)} 個段落 (📡 累計 API 呼叫: {api_call_count} 次)")
                    time.sleep(2)

            # 3️⃣ 結算並紀錄
            if not quota_exhausted:
                new_row = {
                    "股票代碼": stock_code,
                    "年度": year,
                    "公司簡稱": company_name,
                    **stats
                }
                new_analyzed_data.append(new_row)
                processed_keys.add(file_key)

                # 💾 即時追加寫入 Excel
                updated_df = pd.concat([existing_df, pd.DataFrame(new_analyzed_data)], ignore_index=True)
                updated_df.to_excel(master_excel_name, index=False)

        # ==========================================
        # 7. 最終結算與時間累計計時器
        # ==========================================
        end_time = time.time()
        current_run_seconds = end_time - start_time  # 本次執行的秒數

        # 📖 讀取以往累計總用時
        previous_total_seconds = 0.0
        if os.path.exists(time_log_file):
            try:
                with open(time_log_file, 'r', encoding='utf-8') as f:
                    time_data = json.load(f)
                    previous_total_seconds = time_data.get("total_seconds", 0.0)
            except Exception as e:
                print(f"⚠️ 讀取歷史時間記錄檔失敗，重新計時: {e}")

        # ➕ 計算總累計秒數
        total_accumulated_seconds = previous_total_seconds + current_run_seconds

        # 💾 寫回更新後的累計秒數
        try:
            with open(time_log_file, 'w', encoding='utf-8') as f:
                json.dump({"total_seconds": total_accumulated_seconds}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 寫入歷史時間記錄檔失敗: {e}")

        # 格式化本次用時 (分/秒)
        this_mins, this_secs = divmod(current_run_seconds, 60)
        # 格式化歷史累計總用時 (小時/分/秒)
        total_hours, rem_secs = divmod(total_accumulated_seconds, 3600)
        total_mins, total_secs = divmod(rem_secs, 60)

        print(f"\n==================== 📊 本次運算統計看板 ====================")
        print(f"📋 本次新增分析: {len(new_analyzed_data)} 份報告書")
        print(f"📡 API 呼叫總計: {api_call_count} 次")
        print(f"⏱️ 本次分析用時: {int(this_mins)} 分 {int(this_secs)} 秒")
        print(f"⏳ 歷史累計總用時: {int(total_hours)} 小時 {int(total_mins)} 分 {int(total_secs)} 秒")
        print(f"💾 總表已更新至: {master_excel_name}")
        print(f"==========================================================")

        if quota_exhausted:
            print(f"\n⚠️ 提示：因 API 每日總額度耗盡暫停，已完成的檔案皆已安全追加至總表中，下次啟動將自動續傳！")
        else:
            print(f"\n✅ 恭喜！本次選擇的產業檔案全數量化分析完成！")