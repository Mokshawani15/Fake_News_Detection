import requests

# Replace with your News API key
api_key = "62f819f2173c4068b3d9a4644b487d0c"

# Specify the country parameter (e.g., 'us' for the United States)
country = "us"
url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    articles = data["articles"]

    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"URL: {article['url']}")
        print("-" * 50)
else:
    print(f"Error: {response.status_code} - {response.text}")
