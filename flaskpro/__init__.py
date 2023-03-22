import os
import requests
from flask import Flask,render_template,request, jsonify
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_page')
def search_page():
    return render_template('search_page.html')   

@app.route('/search')
def search():
 
    query = request.args.get('query')
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey=9HE15L51KC2VW3UQ'
    response = requests.get(url)
    data = response.json()
    results = []
    try:
        for item in data['bestMatches']:
            result = {
                'symbol': item['1. symbol'],
                'name': item['2. name']
            }
            results.append(result)
        return jsonify(results)
    except:
        return jsonify(results)

        

if __name__ == '__main__':
    app.run()
   



