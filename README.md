# #helladeep
Multi sentiment analysis based on twitter hashtags using a Naive Bayes Classifier

## How to run
1. Clone the repository
2. Run `python predict.py "some text"` inside the directory
3. The results are the emotions followed by the probability of 

[![Demo](https://cdn.rawgit.com/sashankg/-helladeep/master/helladeepdemo.gif)](https://youtu.be/KwLUh6Hh1Wo)

## Background
For the final project of the Stanford Summer Institutes Artificial Intelligence Course, my group decided to create a sentiment analyzer using data from twitter. The idea was that people use hastags to label their tweets anyway, so why not use the hastags as labels for different emotions and the remaining tweet as the body of a thought that had the specified label. Most sentiment analysis projects on the internet are binary. We chose to use the wide gamut of emotions, and inspired by "Inside Out", we chose anger, fear, disgust, joy, and sadness. 

We attempted to use an LSTM neural network, which did not work as intended. Since it was the night before we had to present our project, I wanted to have something to show for the week's worth of work we did. So I implemented this Naive Bayes Classifier that used the twitter data we collected.

## Collecting Data
I was mainly in charge of collecting data. I used the Twitter API to fetch tweets based on search queries for specific hashtags. An interesting idea that we had was to include the emoji in the tweets to maximize the data from each tweet. In order to do this, the emoji had to be converted into the English meaning of the emoji (😂 = Face with tears of joy). I used Emojipedia, which let me search the emoji, and returned its meaning. Emojipedia did not have an API, so I used to Beautiful Soup to extract the data from the HTML.

![Screenshot](https://cdn.rawgit.com/sashankg/-helladeep/master/Screenshot%202016-07-06%20at%207.07.53%20PM.png)

## Classifier
I defined a lexicon using the most used words, so that an input can be vectorized. The weight matrix from the Naive Bayes Classifier is stored in `probabilities.npy`. The classifier runs a softmax algorithm on the input text against the stored weight matrix and outputs the probabilities of the emotion.

