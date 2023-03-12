import os

from flask import Flask,render_template
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    return render_template('search.html')   
@app.route('/about')
def about():
    return render_template('about.html')       

if __name__ == '__main__':
    app.run()
   



