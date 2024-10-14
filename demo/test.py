import requests

def main():
    conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}, 
    {"role": "user", "content": [
        {"type": "audio", "audio": "/disk0/aitogether/xishaojian/audio_lab/audio_resources/hainan.WAV"},
        {"type":"text","text":"说话者语速变化怎么样？"}
    ]},
]
    input = {'conver':conversation}
    res = requests.post("http://127.0.0.1:8602/analyse", json=input)
    # 查看状态码
    print(f"Status Code: {res.status_code}")

    # 查看响应头
    print(f"Headers: {res.headers}")

    # 查看响应体（作为文本）
    print(f"Response Body (Text): {res.text}")
if __name__=='__main__':
    main()