import pandas as pd
import numpy as np
import requests as rq
import bs4
import re
import time
import json
import dropbox
from dropbox.files import WriteMode
from tensorflow.keras.models import load_model

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

dbx = dropbox.Dropbox($YOUR_TOKEN)
dbx_path = '/Aplicativos/pitchfork-reviews/reviews.json'
email_list = [$YOUR_EMAILS]                 

# Write docstrings for these functions
def file_exists(path):
    try:
        dbx.files_get_metadata(path)
        return True
    except:
        return False

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def sentence_to_indices(X, word_to_index, max_review_length, begin_end=False):
    X_indices = np.zeros((1, max_review_length))
    
    sentence_words = X.split()
    if begin_end:
        half = int(max_review_length/2)
        sentence_words = sentence_words[:half] + sentence_words[-half:]
    
    j = 0
    for w in sentence_words[:max_review_length]:
        X_indices[0,j] = word_to_index.get(w, word_to_index['<UNK>'])
        j += 1
  
    return X_indices

# Function to get .json review data
def get_reviews_json(url, size, headers, start=0):
    url_search = url.format(size=size, start=start)
    return rq.get(url_search, headers=headers).json()

# Function to scrape content from review page
def get_and_save_review_content(review_series, save=False):
    url_review = 'https://pitchfork.com'+review_series[0]
    resp = rq.get(url_review)
    # Save HTML
    if save:
        name = review_series[0][16:-1]
        with open("./Reviews/{}.html".format(name), 'w+', encoding='utf-8') as output:
            output.write(resp.text)
    # Retrieve content
    parsed_html = bs4.BeautifulSoup(resp.text)
    description_test = parsed_html.find('div', attrs={'class': 'review-detail__abstract'})
    content_test = parsed_html.find('div', attrs={'class': re.compile(r"contents")})
    # IF first attempt to access page fails, try again until successful
    while (description_test == None) | (content_test == None):
        time.sleep(5)
        resp = rq.get(url_review)
        parsed_html = bs4.BeautifulSoup(resp.text)
        description_test = parsed_html.find('div', attrs={'class': 'review-detail__abstract'})
        content_test = parsed_html.find('div', attrs={'class': re.compile(r"contents")})
    description = parsed_html.find('div', attrs={'class': 'review-detail__abstract'}).get_text().strip()
    content_raw = parsed_html.find('div', attrs={'class': re.compile(r"contents")}).get_text().strip()
    # Removing extra text
    if re.search('\n\n', content_raw) != None:
        content_limit = re.search('\n\n', content_raw).span()[0]
        content = content_raw[:content_limit]
    else:
        content = content_raw
    review = description + " " + content
    return review

# Function to scrape all review pages
def get_new_reviews(size):
    
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Kanguage': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Host': 'pitchfork.com',
        'Referer': 'https://pitchfork.com/reviews/albums/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'
    }
    url = 'https://pitchfork.com/api/v2/search/?types=reviews&hierarchy=sections%2Freviews%2Falbums%2Cchannels%2Freviews%2Falbums&sort=publishdate%20desc%2Cposition%20asc&size={size}&start={start}'
    start = 0
    
    reviews_json = get_reviews_json(url, size, headers)
    reviews_raw_df = pd.json_normalize(pd.json_normalize(reviews_json)['results.list'][0])
    
    new_reviews_df = pd.DataFrame()
    new_reviews_df['url'] = reviews_raw_df['url'].copy()
    new_reviews_df['artists'] = ''
    new_reviews_df['album'] = reviews_raw_df['seoTitle'].copy()
    new_reviews_df['label'] = ''
    new_reviews_df['genres'] = ''
    new_reviews_df['author'] = ''
    new_reviews_df['review'] = ''
    new_reviews_df['rating'] = ''
    new_reviews_df['rounded_rating'] = ''
    new_reviews_df['date'] = pd.to_datetime(reviews_raw_df['pubDate'])

    for i in range(start, size):
        new_reviews_dict = {}
        aux_df = pd.json_normalize(reviews_raw_df.iloc[i,24])
        # Artists
        if reviews_raw_df.iloc[i,0] != []:
            artists = pd.json_normalize(reviews_raw_df.iloc[i,0])['display_name'].str.cat(sep=' ')
            new_reviews_df['artists'][i] = artists
        # Label
        if aux_df.empty == False:
            if pd.json_normalize(aux_df['labels_and_years'][0])['labels'][0] != []:
                label = pd.json_normalize(pd.json_normalize(aux_df['labels_and_years'][0])['labels'][0])['name'].str.cat(sep=' ')
                new_reviews_df['label'][i] = label
        # Genres
        if reviews_raw_df.iloc[i,1] != []:
            genres = pd.json_normalize(reviews_raw_df.iloc[i,1])['slug'].str.cat(sep=' ')
            new_reviews_df['genres'][i] = genres
        # Author
        author = pd.json_normalize(reviews_raw_df.iloc[i,12])['name'].str.cat(sep=' ')
        new_reviews_df['author'][i] = author
        # Review
        review = get_and_save_review_content(new_reviews_df.iloc[i,:])
        new_reviews_df['review'][i] = review
        # Rating
        if aux_df.empty == False:
            rating = aux_df['rating.rating'][0]
            new_reviews_df['rating'][i] = rating
            new_reviews_df['rounded_rating'][i] = round(float(rating))
    
    return new_reviews_df

def compute_prediction(review, review_model, word_to_index, max_review_length):
    review_indices = sentence_to_indices(review, word_to_index, max_review_length, begin_end=True)
    pred = review_model.predict(review_indices)
    pred = np.argmax(pred)
    return pred

def update_counts(diff, count_0, count_1, count_2):
    if diff != 0:
        if abs(diff) == 1:
            count_1 += 1
        elif abs(diff) == 2:
            count_2 += 1
    else:
        count_0 += 1
    return count_0, count_1, count_2  

def clean_review(string):
    string = re.sub("\n", " ", string)
    string = re.sub(" +", " ", string)
    string = string.lower()
    string = re.sub(r"\xa0|\\xbd|\\|\/|\"|\“|\”|\-|\,|\—|\;|\:|\.|\?|\!|\(|\)|\_|\*", " ", string)
    string = re.sub(" +", " ", string)
    string = string.strip()
    return string

def format_review(row, predicted_rating=None):
    date = row['date'].date()
    date_str = date.strftime('%d-%m-%Y')
    rounded_rating = round(float(row['rating']))
    if predicted_rating == None:
        predicted_rating = int(row['predicted_rating'])

    review_formatted = """
    <tr>
        <td style="text-align: center; vertical-align: middle;">{date}</td>
        <td style="text-align: center; vertical-align: middle;"><a href=\"{link}\">{aa}</a></td>
        <td style="text-align: center; vertical-align: middle;">{genres}</td>
        <td style="text-align: center; vertical-align: middle;">{rating}</td>
        <td style="text-align: center; vertical-align: middle;">{rounded_rating}</td>
        <td style="text-align: center; vertical-align: middle;">{predicted_rating}</td>
        <td style="text-align: center; vertical-align: middle;">{difference_ratings}</td>
    </tr>""".format(date=date_str, link='http://pitchfork.com'+row['url'], aa=row['artists']+" - "+row['album'], genres=row['genres'], rating=row['rating'], \
                    rounded_rating=rounded_rating, predicted_rating=predicted_rating, difference_ratings=rounded_rating-predicted_rating)
  
    return (date, review_formatted)

def send_email(email_list, artists, album, url):
    
    link = 'http://pitchfork.com'+url
    aa = artists+' - '+album
    port = 465  # For SSL
    password = $YOUR_PASSWORD
    sender_email = $YOUR_SENDER_EMAIL
    message = MIMEMultipart("alternative")
    message["Subject"] = "Good review released @ Pitchfork ❤"
    message["From"] = sender_email

    # Create the plain-text and HTML version of your message
    text = "There's new good music for you! Enjoy :)"

    html = """\
    <html>
      <body>
        <p>There's new good music for you! Enjoy :)<br><br>
           <a href="{}">{}</a> 
        </p>
      </body>
    </html>
    """.format(link, aa)

    # Turn these into plain/html MIMEText objects
    plain_obj = MIMEText(text, "plain")
    html_obj = MIMEText(html, "html")

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(plain_obj)
    message.attach(html_obj)

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        for email in email_list:
            message["To"] = email
            server.sendmail(sender_email, email, message.as_string())
    
    return True

# Save reviews on .json file
def save_reviews(new_reviews_df, review_model, word_to_index, max_review_length):
    reviews_df = pd.DataFrame()
    new_review_dict = {}
    # Load Dropbox file
    if file_exists(dbx_path):
        dbx.files_download_to_file('./reviews.json', dbx_path, rev=None)
    
    with open("./reviews.json", 'a+') as json_file:
        # OLD REVIEWS (JSON) 
        i = 0
        json_file.seek(0)
        line = json_file.readline()
        while line != '':
            aux = json.loads(line)
            aux_df = pd.DataFrame(aux, index=[i])
            aux_df['date'] = pd.to_datetime(aux_df['date'])
            reviews_df = pd.concat([reviews_df, aux_df])
            i += 1
            line = json_file.readline()
    
        if reviews_df.empty == False:
            new_url = list(set(new_reviews_df['url'])-set(reviews_df['url']))
            difference_df = new_reviews_df[new_reviews_df['url'].isin(new_url)]
        else:
            difference_df = new_reviews_df.copy()

        # NEW REVIEWS
        for idx, rev in difference_df.iterrows():
            new_review_dict['date'] = str(rev['date'].date())
            new_review_dict['url'] = rev['url']
            new_review_dict['artists'] = rev['artists']
            new_review_dict['album'] = rev['album']
            new_review_dict['genres'] = rev['genres']
            new_review_dict['rating'] = rev['rating']
            # Send email
            if float(rev['rating']) >= 8.2:
                send_email(email_list, rev['artists'], rev['album'], rev['url'])
            rounded_rating = round(float(rev['rating']))
            new_review_dict['rounded_rating'] = str(rounded_rating)
            clean_rev = clean_review(rev['review'])

            # Prediction
            predicted_rating = compute_prediction(clean_rev, review_model, word_to_index, max_review_length)
            new_review_dict['predicted_rating'] = str(predicted_rating)
            difference_ratings = rounded_rating - predicted_rating
            new_review_dict['difference_ratings'] = str(difference_ratings)

            json_file.write("{}\n".format(json.dumps(new_review_dict, ensure_ascii=True)))
        json_file.seek(0)
        dbx.files_upload(json_file.read().encode('utf-8'), dbx_path, mode=WriteMode('overwrite'))
        json_file.close()
    
    return True

# Build HTML table with reviews from .json file
def build_reviews():
    reviews_df = pd.DataFrame()
    reviews_formatted = []
    count_0, count_1, count_2 = 0, 0, 0
    dbx.files_download_to_file('./reviews.json', dbx_path, rev=None)
    
    with open("./reviews.json", 'r') as json_file:
        i = 0
        line = json_file.readline()
        while line != '':
            aux = json.loads(line)
            aux_df = pd.DataFrame(aux, index=[i])
            aux_df['date'] = pd.to_datetime(aux_df['date'])
            reviews_df = pd.concat([reviews_df, aux_df])
            count_0, count_1, count_2 = update_counts(int(aux_df['difference_ratings'][i]), count_0, count_1, count_2)
            reviews_formatted.append(format_review(aux_df.iloc[0,:]))
            i += 1
            line = json_file.readline()
        json_file.close()

    # Calculate metrics
    metrics = {}
    count_rev = i
    accuracy_0 = int(round(count_0/count_rev, 2)*100)
    accuracy_1 = int(accuracy_0 + round(count_1/count_rev, 2)*100)
    accuracy_2 = int(accuracy_1 + round(count_2/count_rev, 2)*100)
    misclassification_error = 100 - accuracy_0
    
    metrics['count_rev'] = count_rev
    metrics['accuracy_0'] = accuracy_0
    metrics['accuracy_1'] = accuracy_1
    metrics['accuracy_2'] = accuracy_2
    metrics['misclassification_error'] = misclassification_error

    # Sort reviews_formatted by date
    reviews_formatted = sorted(reviews_formatted, key=lambda x: x[0], reverse=True)

    table = ''
    for date, review in reviews_formatted:
        table = table+review+'\n'
    table = table.strip()

    return table, metrics

