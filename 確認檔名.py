import os
import pandas as pd

# base_floder_path 要輸入使用者的電腦文件路徑
# ================= 參數設定 =================
base_folder_path = r"G:\我的雲端硬碟\參考論文\永續報告書"
excel_path = "company_list.xlsx"
code_column_name = "股票代碼"

# 設定欲保留的目標年份（2023~2025）=> 可自行更改需要年份
TARGET_YEARS = {"2023", "2024", "2025"}

# ================= 1. 選擇要分析的產業 (支援多選 / 全選) =================
if not os.path.exists(base_folder_path):
    print(f"錯誤：找不到資料夾路徑 '{base_folder_path}'")
    exit()

# 取得永續報告書底下的所有產業資料夾
industries = [
    d for d in os.listdir(base_folder_path)
    if os.path.isdir(os.path.join(base_folder_path, d))
]

if not industries:
    print("找不到任何產業資料夾！")
    exit()

print("================ 請選擇要分析的產業 ================")
print("[0] 全選 (分析所有產業)")
for idx, industry in enumerate(industries, 1):
    print(f"[{idx}] {industry}")

print("-" * 52)
print("提示：可輸入多個編號/名稱，用逗號或空格隔開 (例如: 1, 3, 5 或 1 3 5)")
selected_input = input("請輸入產業選擇：").strip()

selected_industries = []

# 檢查是否全選
if selected_input in ["0", "all", "ALL", "a", "A"]:
    selected_industries = industries
else:
    # 替換常見的分隔符號 (逗號、頓號、斜線等) 為半形空格，再分割輸入
    normalized_input = selected_input.replace(",", " ").replace("，", " ").replace("、", " ")
    tokens = normalized_input.split()

    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(industries):
                if industries[idx] not in selected_industries:
                    selected_industries.append(industries[idx])
        elif token in industries:
            if token not in selected_industries:
                selected_industries.append(token)

if not selected_industries:
    print("\n輸入無效或找不到任何匹配的產業，程式結束。")
    exit()

print(f"\n已選擇產業 ({len(selected_industries)} 個)：【{', '.join(selected_industries)}】")
print("開始跨產業分析...\n")

# ================= 2. 讀取公司清單 (company_list.xlsx) =================
valid_stock_codes = set()
if os.path.exists(excel_path):
    try:
        df_company = pd.read_excel(excel_path)
        possible_cols = [col for col in df_company.columns if "代碼" in str(col) or "代號" in str(col)]
        target_col = possible_cols[0] if possible_cols else code_column_name

        if target_col in df_company.columns:
            # 強制轉字串並去空白
            valid_stock_codes = set(df_company[target_col].astype(str).str.strip())
            print(f"已成功載入公司清單，共 {len(valid_stock_codes)} 筆股票代碼。\n")
        else:
            print(f"警告：公司清單中找不到欄位 '{code_column_name}'，將跳過股票代碼比對。\n")
    except Exception as e:
        print(f"讀取 Excel 失敗: {e}，將跳過股票代碼比對。\n")
else:
    print(f"提示：未找到 '{excel_path}'，將跳過股票代碼比對。\n")

# ================= 3. 掃描選定產業目錄下的 PDF 檔案檔名 =================
error_list = []        # 格式/代碼錯誤清單
out_of_range_list = []   # 多出的年份（非 2023~2025 年）清單
total_pdf_count = 0

# 逐一對選定的產業資料夾進行掃描
for industry in selected_industries:
    target_folder_path = os.path.join(base_folder_path, industry)

    for root, dirs, files in os.walk(target_folder_path):
        for filename in files:
            # 跳過暫存檔或隱藏檔
            if filename.startswith("~$") or filename.startswith("."):
                continue

            # 只針對 PDF 檔進行檢查
            if filename.lower().endswith('.pdf'):
                total_pdf_count += 1
                name_without_ext, _ = os.path.splitext(filename)

                # 取得相對路徑（包含產業名稱，例如：水泥工業/信大水泥/1109_2022.pdf）
                rel_path = os.path.relpath(os.path.join(root, filename), base_folder_path)

                # --- 檔名格式細化檢測 ---
                if "_" not in name_without_ext:
                    error_list.append((rel_path, filename, "未使用底線 '_'"))
                else:
                    parts = name_without_ext.split("_")
                    if len(parts) != 2:
                        error_list.append((rel_path, filename, "檔名格式全錯 (包含多個或位置不對的底線)"))
                    else:
                        stock_code, year = parts[0].strip(), parts[1].strip()

                        code_err = False
                        # 1. 檢查股票代碼比對
                        if valid_stock_codes and stock_code not in valid_stock_codes:
                            error_list.append((rel_path, filename, f"股票代碼錯誤 (代碼 '{stock_code}' 不在公司清單中)"))
                            code_err = True

                        # 2. 檢查年份格式與範圍
                        if not year.isdigit() or not (3 <= len(year) <= 4):
                            if not code_err:  # 代碼無誤才標示年度格式錯誤
                                error_list.append((rel_path, filename, f"年度格式錯誤 (年度 '{year}' 非有效數字年份)"))
                        else:
                            # 3. 檢查是否屬於多出的非目標年份 (非 2023~2025)
                            if year not in TARGET_YEARS:
                                out_of_range_list.append((rel_path, filename, year))

# ================= 4. 直接輸出總體分析結果 =================
print("=" * 85)
print(f"跨產業 PDF 檔名總體分析結果 (共檢查 {len(selected_industries)} 個產業)")
print(f"掃描 PDF 總數：{total_pdf_count} 個")
print(f"格式錯誤檔案數：{len(error_list)} 個")
print(f"多出年份檔案數：{len(out_of_range_list)} 個 (非 2023~2025 年)")
print("=" * 85)

# 輸出多出的年份檔案 (2021, 2022 等)
if out_of_range_list:
    print("\n[ 多出/非目標區間的年份檔案清單 (如 2021、2022) ]\n")
    print(f"{'相對路徑與檔名':<55} | {'檔案年份'}")
    print("-" * 85)
    for path, fname, yr in out_of_range_list:
        print(f"{path:<55} | {yr} 年")
    print("-" * 85)

# 輸出格式錯誤檔案
if error_list:
    print("\n[ 格式錯誤檔案清單 ]\n")
    print(f"{'相對路徑與檔名':<55} | {'錯誤原因'}")
    print("-" * 85)
    for path, fname, err_reason in error_list:
        print(f"{path:<55} | {err_reason}")
    print("-" * 85)

if not error_list and not out_of_range_list:
    print("\n太棒了！所選產業下所有 PDF 檔名均符合格式，且年份皆在 2023~2025 年區間內！")