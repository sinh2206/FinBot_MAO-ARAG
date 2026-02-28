import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load các biến môi trường từ file .env (nơi chứa GEMINI_API_KEY của bạn)
load_dotenv()

# Lấy API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Lỗi: Không tìm thấy GEMINI_API_KEY trong file .env!")
    exit()

# 2. Cấu hình thư viện với API Key của bạn
genai.configure(api_key=api_key)

print("🔍 Đang kiểm tra danh sách model khả dụng...")
print("=" * 60)

# 3. Duyệt qua danh sách các model và in ra kết quả
try:
    models = genai.list_models()
    count = 0
    for m in models:
        # Chỉ lọc ra các model hỗ trợ tạo văn bản (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Tên Model : {m.name.replace('models/', '')}")
            print(f"📝 Mô tả     : {m.description}")
            print("-" * 60)
            count += 1
            
    print(f"🎉 Tuyệt vời! API Key của bạn có quyền truy cập vào {count} model tạo văn bản.")
except Exception as e:
    print(f"❌ Có lỗi xảy ra khi gọi API: {e}")