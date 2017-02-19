import nltk,pickle,re
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#import string
import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
import operator

stemmer = LancasterStemmer()

def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

def processTweet(tweet):
    # process the tweets
    
    hashtags = []

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    hashtags = re.findall(r"#(\w+)", tweet)
    tweet = re.sub(r'#([^\s]+)'," ", tweet)                 #may nedd to change
    #trim
    tweet = tweet.strip('\'"')
    return tweet,hashtags
    
def getFeatureVector(tweet):
    stopWords = list(stopwords.words("english"))
    stopWords.append('AT_USER')
    stopWords.append('URL')
    
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(stemmer.stem(w).lower())
    return featureVector    

def find_features(document,word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features
    
def get_required_list():
    
    word_features = []
    data = pd.read_csv(".......Tweets Location........")
    
    df = pd.DataFrame(columns=["Word"])
    df.set_index("Word",inplace = True)
    
    required_list  = []
    
    for twt in data["text"].values:
        
            processed_twt,hashtags = processTweet(twt)
            features = getFeatureVector(processed_twt)
            for i in features:
                word_features.append(i)
                
            word_category_list = [(features, category)
                                 for category in hashtags ]
                                 
            required_list += word_category_list

    return required_list,word_features
    
def train_classifiers():
     
    documents,word_features = get_required_list()
    
    save_word_features = open(".......Location........")
    pickle.dump(word_features, save_word_features)
    save_word_features.close()
    
    featuresets = [(find_features(rev,word_features), category) for (rev, category) in documents]
    print (featuresets)
    
    break_point = int((95/100)*len(featuresets))
    
    training_set = featuresets[:break_point] #need to change
    #
    ## set that we'll test against.
    testing_set = featuresets[break_point:] # need to change
    
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    
    save_classifier = open(".......Location........","wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()
    
    print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
    classifier.show_most_informative_features(15)
    
def custom_predict(new_twt):
    
    di = {}
    predictions = []

    df = pd.read_pickle(".......Location........")
    processed_twt,original_hashtags = processTweet(new_twt)
    filtered_words = getFeatureVector(processed_twt)
    
    for f in df.columns:
        value = 1
        for r in filtered_words:
            if r in df.index:
                value = value + df.loc[r,f]
        di[f] = value * df.loc["probability",f]
    
    sorted_di = sorted(di.items(), key=operator.itemgetter(1))
    
    for i in range(1,6):
        predictions.append(sorted_di[-i][0])
        
    return original_hashtags,predictions
    
def train_custom_naive_bayes():
    
    e = 0.0001
#    
    data = pd.read_csv(".......Tweets Location........")
##    
    break_point = int((95/100)*len(data["text"].values))
    print (break_point)
    
    df = pd.DataFrame(columns=["Word"])
    df.set_index("Word",inplace = True)
    
    for twt in data["text"][:break_point].values:
        
        check_hash_train = re.findall(r"#(\w+)", twt)
        if check_hash_train:
        
            processed_twt,hashtags = processTweet(twt)
            filtered_words = getFeatureVector(processed_twt)
            
            for i in hashtags:
                if i not in df.columns:
                    df[i] = e
        
        
            for j in filtered_words:
                if j in df.index:
                    for k in hashtags:
                        df.loc[j,k] += 1
                else:
                    for t in df.columns:
                        if t in hashtags:
                            df.loc[j,t] = 1
                        else:
                            df.loc[j,t] = e            
    total_value = 0
    
    for f in df.columns:
        sum = 0
        for r in df.index:
            if r != "sum":
                sum += df.loc[r,f]
        total_value += sum
        df.loc["sum",f] = sum
        
    for f in df.columns:
        for r in df.index:
            if r != "sum":
                df.loc[r,f] = df.loc[r,f]/df.loc["sum",f]
            else:
                df.loc["probability",f] = df.loc[r,f]/total_value
    
    df.to_pickle(".......Location........")

    
    total_predictions = 0
    correct_predictions = 0
    
    for twt in data["text"][break_point:].values:

        check_hash_test = re.findall(r"#(\w+)", twt)
        if check_hash_test:
            p = str(predict(twt))
            original_hashtags,predicted_hashtags = custom_predict(twt)
            total_predictions += 1
            if p in predicted_hashtags:
                correct_predictions += 1
    print("Custom Classifier accuracy percent:",correct_predictions/total_predictions)   
        

def predict(new_twt):
    
    word_features5k_f = open(".......Location........", "rb")
    word_features = pickle.load(word_features5k_f)
    word_features5k_f.close()

    open_file = open(".......Location........", "rb")
    classifier = pickle.load(open_file)
    open_file.close()
    
    processed_twt,original_hashtags = processTweet(new_twt)
    filtered_words = getFeatureVector(processed_twt)
    features = find_features(filtered_words,word_features)
    return classifier.classify(features)

#def predict_with_custom_naive(new_twt):

    
    