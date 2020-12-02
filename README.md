# Pitchfork Reviews

### Modeling

This project is an attempt to predict a Pitchfork review rating (ranging from 0 to 10 with steps of 0.1) using an RNN trained on the review's content (with Tensorflow/Keras). For that, I initially collected all reviews from Pitchfork's website, cleaned the data, used GloVe embeddings with 50 dimensions to transform words into vectors, and finally trained several models, choosing the one with the best performance.

Since I'm using a regular notebook with 8GB RAM and no GPU, training a deep RNN on it proved impossible. So, I resorted to the free GPU available at Google Colab (http://colab.research.google.com/) to train my models. 
The initial architecture design for the neural network is depicted below:

![Initial architecture](https://github.com/rafael-siqueira/pitchfork/blob/master/images/Architecture.png)

At first, I trained a regression RNN with a sigmoid activation function and MSE as its loss function, changing my labels from a 0-10 scale to a 0-1 (by simply dividing them by 10). The results obtained were not great.   

Since I presumably didn't have enough training data to be able to achieve the granularity demanded by a continuous (albeit bounded) output, I tried to reframe the problem to a multi-class classification one, using a softmax activation with 101 output nodes (for the ratings ranging from 0 to 10 with steps of 0.1) and categorical cross-entropy loss. Once again, the results were not exciting.  

So, I simplified the problem again, rounding the labels to integers from 0 to 10. Therefore, I still used a softmax activation, but now with only 11 output nodes. This approach seemed promissing by the initial results obtained and so I started tweaking hyperparameters to try to enhance accuracy. After several tests, I obtained the following model evolution, expressed in the table below:

![Model evolution](https://github.com/rafael-siqueira/pitchfork/blob/master/images/Models_Results.png)

After model 3, I continued increasing complexity by having 4 LSTM layers and the results started getting worse. **Therefore, since model 3 had good accuracy and MSE values for the dev set, in comparison to the other models, I decided to use it on the test set and the results were not terrible. So, I decided to use model 3 in production, for now.**
Also, as expected, since there are very few training examples with low ratings (0-4), the RNN was not able to learn very well the particularities of these type of reviews and doesn't predict much on this lower range (I only evaluated maximum and minimum predicted ratings for model 3).

The Jupiter notebook `Pitchfork_vgit.ipynb` contains the code for the process described above, with model 3's implementation. In order to be able to use the model trained or train new models, you need to uncompress the GloVe and model .rar files.

### Deploying

Having the model, I implemented a very simple webpage (didn't bother much with styling for now) which contains the most recent reviews, their predicted ratings and accuracy metrics. The webpage is updated daily to include new reviews. And whenever a new review is above a certain threshold (8.2), an e-mail is sent to me and my brother with the album's review link.
The metrics tracked and shown in the webpage are the following:
- Accuracy: rate of reviews with predicted ratings equal to real rounded ratings
- Accuracy±1: rate of reviews with predicted ratings equal to or off by 1 point from real rounded ratings
- Accuracy±2: rate of reviews with predicted ratings equal to or off by 2 points from real rounded ratings
- Misclassification Error: rate of reviews with predicted ratings different from real rounded ratings (1-Accuracy)

The scripts `app.py`, `monitor.py`, `review_utils.py` and `wsgi.py` are the ones running and maintaining the website. It was implemented with Flask, Heroku and Dropbox. Flask for creating the web app, Heroku to host it and Dropbox for storing the .json file with the reviews' information. As next steps, I intend to try out different architectures/models, implement a DB to store the information and make the website look a little better ;)

The webpage address is https://pitchfork-reviews.herokuapp.com/  
Since it is run on free Heroku's dynos, it is shut down if there is no traffic for a while. Therefore, load time might be around 30 seconds which is the time to restart the dyno and unidle the process.
