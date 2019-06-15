from flask import Flask, render_template, request
from sklearn.externals import joblib
import os

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/bike_demand', methods=['POST', 'GET'])
def bike_demand():
    # get the parameters
    Month = int(request.form['Month'])
    Day = int(request.form['Day'])
    Hour = int(request.form['Hour'])

    # load the model and predict
    model = joblib.load('model/bixi-model-bike-demand-hourly.pkl')
    prediction = model.predict([[Month, Day, Hour]])
    count = prediction.round(0)

    return render_template('results.html',
                           Month = int(Month),
                           Day = int(Day),
                           Hour = int(Hour),
                           count = count
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug = True)
