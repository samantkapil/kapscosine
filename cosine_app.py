from flask import Flask, render_template, request
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

app = Flask(__name__)


@app.route('/')
def cosine_input():
    return render_template('cosine.html')

@app.route('/cosine' , methods = ['post'])
def cosine():
    count_vector_data = CountVectorizer(stop_words='english')
    text1 = request.form.get('text1')
    text2 = request.form.get('text2')

    document = [text1,text2]

    sparse_matrix = count_vector_data.fit_transform(document)

    df= pd.DataFrame(sparse_matrix.todense(), columns=count_vector_data.get_feature_names(),
                index=['text1','text2'])

    output = cosine_similarity(df,df)
    print(output)
    if output[0][1]>=0.8:
        ans = 'Similar'
    else:
        ans = 'not similar'
    return render_template('cosine_op.html', cosine_op= f'the word is {ans}' )



if __name__ == '__main__':          
    app.run(host='0.0.0.0', port=8080)