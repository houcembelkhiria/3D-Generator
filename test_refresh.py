import requests
import json

url = "http://localhost:6400/mcp"
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "refresh_unity",
        "arguments": {}
    }
}
response = requests.post(url, json=payload)
print(response.text)
