# Sentiment Analysis on Tourist Accommodation Reviews
I developed a Sentiment Analysis Model for Tourist Accommodation Reviews using Natural Language Processing Techniques.
The best model achieved a 86.3% accuracy and an f1 with macroaveraging of 64.2% 

## Code and Resources
* **Python version:** 3.7
* **Packages:** gensim, nltk, textblob, re, sklearn, matplotlib, seaborn, pandas, numpy
* **Dataset:** https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews
* **Bibliography:**
  1. Jurafsky, D. and J.H. Martin Speech and language processing. (2021; 3rd draft ed.) Chapter 4 Naive Bayes and Sentiment Classification
  2. Ozen, I. A. (2021). Tourism products and sentiment analysis. In C. Cobanoglu, S. Dogan, K. Berezina, & G. Collins (Eds.), Advances in Hospitality and Tourism Information Technology (pp. 1–44). USF M3 Publishing. https://www.doi.org/10.5038/9781732127586

## Project Overview
* Developed a model to predict the sentiment of a textual Tourist Accommodation Review
* Showed the keywords that are used in negative and positive reviews.
* Developed different pre-processing strategies using NLP tools. 
* Applied different machine learning models.
* Used pre-trained lexicon based-models for sentiment analysis
* Evaluated the best combination of pre-processing and ML models and the pre-trained model
* The best model achieved an accuracy of 86.3% and an f1 with macroaveraging of 64.2% 
* Found that it is feasible to predict positive and negative reviews, but not so much the neutral ones.

## Short Backgroung
Tourists always have things to say about their holidays, whether they are good or bad. Tourist online platforms take advantage of this and encourage tourists to leave reviews about the tourism products they consumed while on holiday. These reviews are considered as a rich data source of information both for the tourism industry and for future travellers. 

The number of reviews that exist on a single platform is too high to be analysed manually by a team of people. Hence, Natural Language Processing (NLP) techniques together with Machine Learning methods can help in the analysis process. The tourism businesses, such as accommodation providers, should take advantage of these technologies and the existing publicly available reviews datasets to create sentiment analysis models that could help them position themselves ahead in this market.


## Objectives
Create a sentiment analysis model for the tourism industry, especially for the accommodation sector which includes hotels, hostels, B&Bs and holidays resorts among others.
The model will have two main functions:

1.	It will potentially show the keywords that are used in negative and positive reviews and 
2.	It will serve as a predictor for the positivity or negativity of future textual reviews from future customers.

## Evaluation metric
I look both at the accuracy and the f1 with microavaraging due to an unbalanced dataset.

## EDA
After feature engineering- transforming ranking (1-5) to sentiment (Neg, Nue, Pos), I looked at the sentiment distributions:
![](https://github.com/CarolinaKra/SentimentAnalysisHotelReviews/blob/main/images/sentimentDistribution.png)

## Pre-Processing Strategies
1. Simple clean and bag of words (tf-idf) for the baseline model
2. Simple clean, adding similar words of adjectives in the text using embedings and lemmatisation on the text and then proceed with a bag of words (tf-idf), while excluding "stop words". This was used for different ML models.
3. Simple clean and lemmatisation, removal of "stop words" and counting only once each word that appears in the review, extracting only a few of the most common words in the full text. This was used for different ML models.
4. No pre-process for lexicon-based models for sentiment analysis.

## Modelling
* The Baseline Model used is the Gausian Naive Bayes model 

After the pre-processing strategy no 2, I applied the models:
* Gaussian Naive Bayes
* Multinomial Naive Bayes 
* Decision Tree
* Suport Vector Machine
* Weighted Support Vector Machine

After the pre-processing strategy no 3, I applied the models:
* Naive Bayes
* Decision Tree
* Support Vector Machine

For the unprocessed text, strategy no 4, I applied the following pre-trained, lexicon-based models for sentiment analysis:
* Textblob
* SentimentIntensityAnalyzer from nltk 

## Evaluation of all the models
The final model accuracy and f1 with micro-averaging and for each class are displayed in the following table:
![alt text](https://github.com/CarolinaKra/SentimentAnalysisHotelReviews/blob/main/images/EvaluationTable.png)

The best model is the one which uses the strategy no 2 and SVM, the model results could be visualised in the following confusion matrix:
![alt text](https://github.com/CarolinaKra/SentimentAnalysisHotelReviews/blob/main/images/svmConfMatrix.png)

From the confusion matrix, we can see that the model is good at predicting Positive and Negative Reviews but not so good at predicting the Neutral reviews.

However, this model is less interpretable and hard to get insights about the keywords that have main influence in the sentiment classification. On the other hand, using strategy no 3 and the Naive Bayes model, I could get the main features that have such effect, these are:
![alt text](https://github.com/CarolinaKra/SentimentAnalysisHotelReviews/blob/main/images/NLPimportantFeatures.png)

It is interesting to see that some words are obvious to be in a specific class such as: 'perfect', 'loved', 'superb' as keywords for positive reviews and words such as 'worst','poor','dirty' for negatives but some words that we wouldn't expect, are actually key to classify a class. For example: 'finally' appears to be one of the keywords for negative reviews, meaning that people wrote an experience and commented how it finished.

## Final Conclusions
* The combination of using embeddings, lemmatisation and bag of words together with SVM, is the best text classifier model for sentiment analysis. 
* Even though it showed a high percentage accuracy (86.3%), it perform poorly to classify the reviews which were classified as Neutral.
* This model fulfils the objective: to create a sentiment analysis model for the tourism industry, especially for the accommodation sector.
* This model will serve as a predictor for the positivity or negativity of future textual reviews from future customers.
* However, this model does not show the keywords that are used in negative and positive reviews, as it is hard to be interpreted.
* But, the model made with feature extraction and Naive Bayes, gave a list of the most informative features that affect the sentiment analysis classification. 
* This project has demonstrated that the combination of several techniques can help achieve the final objective which was to give NLP tools that can help a business in the tourism industry position itself ahead in the market. 
* Furthermore, the same preprocessing and model could be applied for a similar dataset for a different sector within the tourism industry, such as excursions and attractions, or for other sentiment analysis tasks, such as movie reviews. 





