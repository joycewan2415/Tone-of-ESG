# Tone-of-ESG
**主題類別**  
Text Mining, Data Preprocess, Large Language Model, ESG reports, Sentiment Classification

**分析工具**  
/n Python(google.generativeai, pymupdf, rapidocr_onnxruntime, pandas, re, json, time, gc)

**主要目標**  
計算台灣上市公司中文永續報告書的揭露語調分數(Tone)，並搭配ESG主題分類。其中，**語調分數(Tone)** 介於-1~1，有三種情緒，分別為**正向、負向、中立**。愈接近0為愈中立，愈接近1為愈正向，愈接近-1則為愈負向。

**研究步驟**  
Part 1:蒐集台灣上市公司的中文永續報告書  
(1)前往ESG數位平台進行人工下載並存入Google雲端中

Part 2:分別計算各永續報告書的語調分數(Tone)--使用語調分析模型  
(1)整理好公司股票代碼及簡稱：將代碼存在list中  
(2)使用Gemini AI進行程式碼生成，下達精確的指令，透過數次的校正修改與增加所需內容，最後生成出一份完整可運行之程式碼<br>
(3)申請Google AI studio中的API KEY  
(4)逐一擷取PDF文字(使用pymupdf與rapidocr_onnxruntime)<br>
(5)json斷句：用 [。！？] 切割成單句，再每兩句合併為一分析單位  
(6)過濾雜訊：清除永續報告書中與ESG分析無關之目錄、地名通訊錄  
(7)ESG語調分析模型  
-(7-1) 情緒分類模型：將本文分類為**正向、負向、中立**  
-(7-2) 主題分類模型：將文本分類為**環境(E)、社會(S)、治理(G)、非ESG**  
(7)紀錄結果：使用pandas套件，將結果存成dataframe格式。  

【輸出結果】  
2023年至2025年知各公司永續報告書語調分數(Tone)結果，可參考  

共有12種分類結果(情緒分類3種 x 主題分類4種)，欄位說明如下：<br>
E_neutral：環境面中立情緒(句數)<br>
E_positive：環境面正向情緒(句數)<br>
E_negative：環境面負向情緒(句數)<br>
S_neutral：社會面中立情緒(句數)<br>
S_positive：社會面正向情緒(句數)<br>
S_negative：社會面負向情緒(句數)<br>
G_neutral：治理面中立情緒(句數)<br>
G_positive：治理面正向情緒(句數)<br>
G_negative：治理面負向情緒(句數)<br>
non_neutral：非ESG中立情緒(句數)<br>
non_positive：非ESG正向情緒(句數)<br>
non_negative：非ESG負向情緒(句數)<br>
