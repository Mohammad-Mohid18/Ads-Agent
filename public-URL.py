from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Relative file path saved in Supabase
file_path = "renders/c9c4e25e-b341-46d5-8b4e-4288b543c22c_v1.mp4"
bucket_name = "ad-assets"

# Option A: Get Public URL (If bucket is public)
preview_url = supabase.storage.from_(bucket_name).get_public_url(file_path)

# Option B: Get Signed URL (If bucket is private - valid for 1 hour / 3600 secs)
# signed_res = supabase.storage.from_(bucket_name).create_signed_url(file_path, 3600)
# preview_url = signed_res.get("signedURL")

print("Preview URL:", preview_url)
