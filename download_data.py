"""Download dataset from OneDrive shared link."""
import urllib.request
import base64
import os

os.makedirs("data", exist_ok=True)

# Encode the sharing URL for the OneDrive API
share_url = "https://1drv.ms/x/c/d29719ef2bac03e2/IQB0JvLEpj9kTZq0pY_IG1SpAUNfOISHc9IAUNG2VGbz4EI?e=gUd4SP"
encoded = base64.urlsafe_b64encode(share_url.encode()).decode().rstrip("=")
api_url = f"https://api.onedrive.com/v1.0/shares/u!{encoded}/root/content"

print(f"Trying API URL: {api_url[:80]}...")
req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
try:
    resp = urllib.request.urlopen(req, timeout=30)
    data = resp.read()
    with open("data/dataset.xlsx", "wb") as f:
        f.write(data)
    print(f"SUCCESS: Downloaded {len(data)} bytes to data/dataset.xlsx")
except Exception as e:
    print(f"API approach failed: {e}")
    
    # Fallback: direct download URL
    direct_url = "https://onedrive.live.com/download?resid=D29719EF2BAC03E2%21sc4f226743fa64d649ab4a58fc81b54a9&authkey="
    print(f"Trying direct URL...")
    req2 = urllib.request.Request(direct_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp2 = urllib.request.urlopen(req2, timeout=30)
        data2 = resp2.read()
        with open("data/dataset.xlsx", "wb") as f:
            f.write(data2)
        print(f"SUCCESS: Downloaded {len(data2)} bytes to data/dataset.xlsx")
    except Exception as e2:
        print(f"Direct approach also failed: {e2}")
        print("Please download manually and place at: data/dataset.xlsx")
