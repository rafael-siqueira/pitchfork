from flask import Flask
from review_utils import *

# Creating Flask object
app = Flask(__name__)

# Python decorator: what URL will trigger the function below
@app.route('/')
def main_page():
    table, metrics = build_reviews()

    html = """
    <html>
        <head>
            <font face="Rockwell">
            <h1>Pitchfork Reviews</h1>
        </head>
        <body>
            <font face="Arial">
            <table border="1">
                <thead>
                    <tr>
                        <th scope="col">Number of Reviews</th>
                        <th scope="col">Accuracy</th>
                        <th scope="col">Accuracy±1</th>
                        <th scope="col">Accuracy±2</th>
                        <th scope="col">Misclassification Error</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align: center; vertical-align: middle;">{}</td>
                        <td style="text-align: center; vertical-align: middle;">{}%</td>
                        <td style="text-align: center; vertical-align: middle;">{}%</td>
                        <td style="text-align: center; vertical-align: middle;">{}%</td>
                        <td style="text-align: center; vertical-align: middle;">{}%</td>
                    </tr>
                </tbody>
            </table>
            </html>
            <br>
            <br>
            <table border="1">
                <thead>
                    <tr>
                        <th scope="col">Date</th>
                        <th scope="col">Artists - Album</th>
                        <th scope="col">Genre</th>
                        <th scope="col">Rating</th>
                        <th scope="col">Rounded Rating</th>
                        <th scope="col">Predicted Rating</th>
                        <th scope="col">Difference in Ratings</th>
                    </tr>
                </thead>
                <tbody>
                {}
                </tbody>
            </table>
        </body>
    </html>""".format(metrics['count_rev'], metrics['accuracy_0'], metrics['accuracy_1'], metrics['accuracy_2'], metrics['misclassification_error'], table)

    return html

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    
