import xgboost
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import librosa
import numpy as np


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from pytube import YouTube
from pydub import AudioSegment
import youtube_dl
import os



# # Model

# In[ ]:


df = pd.read_csv(r'data\Data\features_3_sec.csv')

# df = df.drop(['filename'], axis=1)

# df = df[['chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','label']]

df = df[['chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','tempo','label']]
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


df['label'] =  label_encoder.fit_transform(df['label'])

print(label_encoder.classes_)

y = df[['label']]
X = df[df.columns.difference(['label'])]


## split both X and y using a ratio of 70% training - 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(len(X_train), len(X_test), len(y_train), len(y_test))

xgb = xgboost.XGBClassifier(n_estimators=1000,enable_catergorical=True,learning_rate=0.05)
xgb.fit(X_train, y_train)

## make predictions on the test portion (predict the labels of the rows from the test portion of X)
predictions = xgb.predict(X_test)

target_name = ['blues', 'classical', 'country', 'disco', 'hiphop' ,'jazz' ,'metal', 'pop','reggae' ,'rock']


print(classification_report(y_test, predictions, target_names=target_name))
## can also output the confusion matrix
# cm = confusion_matrix(y_test, predictions)
# print(cm)

print("Accuracy: " ,metrics.accuracy_score(y_test, predictions))

cols_when_model_builds = xgb.feature_names_in_


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

def find_sim(data):
    placeHoldername = 'test'
    data['filename'] = placeHoldername

    df_sim = pd.read_csv(r'data\Data\features_30_sec.csv')

    df_sim = df_sim[['filename','chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','tempo','label']]


    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df_sim['label'] = df_sim['label'].astype("string")
    df_sim['label'] =  label_encoder.fit_transform(df_sim['label'])




    combined_df = pd.concat([df_sim, data], ignore_index=True)

    combined_df = combined_df.set_index('filename')

 
    labels = combined_df[['label']]

    
    scaled = preprocessing.scale(combined_df)
    similarity = cosine_similarity(scaled)
    sim_df_labels = pd.DataFrame(similarity)
    sim_df_names = sim_df_labels.set_index(labels.index)
    sim_df_names.columns = labels.index

    series = sim_df_names[placeHoldername].sort_values(ascending=False)
    series = series.drop(placeHoldername)
    return series.head(3)





# In[ ]:


def find_pred(data, features, predicted_feature):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    features = features.drop(['filename'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, data[predicted_feature], test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)


    prediction = model.predict(features)

    return prediction[0]


# In[ ]:


def confidence_score(proba):
    from collections import Counter
    confi = {}
    i=0
    for val in proba[0]:
        rounded = round(val *100,2)
        confi[target_name[i]] = rounded
        i = i+1

    k = Counter(confi)
    
    high = k.most_common(3) 
    
    # for i in high:
    #     print(i[0]," :",i[1]," ")

    return high


# # Extract Features

# In[ ]:


def extract_features(file):
    y, sr = librosa.load(file)

    chroma_sft_mean =  np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_sft_var =  librosa.feature.chroma_stft(y=y, sr=sr).var()

    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()

    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()

    spectral_bandwith_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwith_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()

    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()

    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y=y).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y=y).var()


    harmony_mean = librosa.effects.harmonic(y).mean()
    harmony_var = librosa.effects.harmonic(y).var()

    tempo = librosa.feature.tempo(y=y, sr=sr)[0]

    
    print("----------------------------------------------------------------------------")

    print("chroma_sft_mean: ",chroma_sft_mean)
    # print("chroma_sft_var: ",chroma_sft_var)
    print("rms_mean: ",rms_mean)
    # print("rms_var: ",rms_var)
    print("spectral_centroid_mean: ",spectral_centroid_mean)
    # print("spectral_centroid_var: ",spectral_centroid_var)
    print("spectral_bandwith_mean: ",spectral_bandwith_mean)
    # print("spectral_bandwith_var: ",spectral_bandwith_var)
    print("rolloff_mean: ",rolloff_mean)
    # print("rolloff_var: ",rolloff_var)
    print("zero_crossing_rate_mean: ",zero_crossing_rate_mean)
    # print("zero_crossing_rate_var: ",zero_crossing_rate_var)
    print("harmony_mean: ",harmony_mean)
    # print("harmony_var: ",harmony_var)
    print("tempo: ",tempo)

    print("----------------------------------------------------------------------------")

    features = pd.DataFrame({'chroma_stft_mean':[chroma_sft_mean],'chroma_stft_var':[chroma_sft_var],'rms_mean':[rms_mean],'rms_var':[rms_var],'spectral_centroid_mean':[spectral_centroid_mean],
                             'spectral_centroid_var':[spectral_centroid_var],'spectral_bandwidth_mean':[spectral_bandwith_mean],'spectral_bandwidth_var':[spectral_bandwith_var],
                             'rolloff_mean':[rolloff_mean],'rolloff_var':[rolloff_var],'zero_crossing_rate_mean':[zero_crossing_rate_mean],'zero_crossing_rate_var':[zero_crossing_rate_var],
                             'harmony_mean':[harmony_mean],'harmony_var':[harmony_var],'tempo':[tempo],})
    

    features = features.reindex(columns=cols_when_model_builds)

    # features = features.reshape(-1,1)
    return features



# # Search Youtube

# In[ ]:


def search(query):
    from pytube import YouTube
    from googleapiclient.discovery import build

    api_key = 'AIzaSyCghPkifWFcLs_iN5CCvLIlQwvWBXxIxxY'

    # Initialize the YouTube Data API
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Perform a video search
    search_response = youtube.search().list(
        q=query,
        type='video',
        part='id,snippet',
        maxResults=1 
    ).execute()

    # Iterate through the search results and get video information
    for search_result in search_response.get('items', []):
        video_id = search_result['id']['videoId']
        # video_title = search_result['snippet']['title']
        video_url = f'https://www.youtube.com/watch?v={video_id}'

        print(f'Video Ulr: {video_url}')
        url = video_url

    video = YouTube(url)

    stream = video.streams.filter(only_audio=True).first()
    stream.download(filename=f"musicaudio.mp3")
    sound = AudioSegment.from_file(r"musicaudio.mp3")
    start_time = 0  
    end_time = 30 * 1000

    audio_segment = sound[start_time:end_time]

    audio_segment.export(r"music/downloaded/musicaudio.mp3", format="mp3")


# # Chatbot(Orpheus)

# In[ ]:


import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

patterns = [
    [{"LOWER": "hello"}],
    [{"LOWER": "hi"}],
    [{"LOWER": "how"}, {"LOWER": "are"}, {"LOWER": "you"}],
    [{"LOWER": "find"}, {"LOWER": "this"}, {"LOWER": "song"}, {"LOWER": "but"}, {"LOWER": {"REGEX": ".*"}}],
    [{"LOWER": "find"}, {"LOWER": "similiar"}, {"LOWER": "songs"}],

]

responses = {
    "greetings": [
        [{"LOWER": "hello"}],
        [{"LOWER": "hi"}],
    ],
    "inquiries": [
        [{"LOWER": "how"}, {"LOWER": "are"}, {"LOWER": "you"}],
    ],

    "find_increased": [
        [{"LOWER": "find"}],
    ],

    "find_sim": [
        [{"LOWER": "search"},{"LOWER": "for"},{"LOWER": "similiar"},{"LOWER": "songs"}],
        [{"LOWER": "search"},{"LOWER": "for"},{"LOWER": "songs"},{"LOWER": "like"},{"LOWER": "this"}],
    ],

    "like": [
        [{"LOWER": "i"}, {"LOWER": "like"},  {"LOWER": {"REGEX": ".*"}}],
        [{"LOWER": "i"}, {"LOWER": "love"},  {"LOWER": {"REGEX": ".*"}}],
        [{"LOWER": "i"}, {"LOWER": "want"},  {"LOWER": {"REGEX": ".*"}}],
        [{"LOWER": "i"}, {"LOWER": "need"},  {"LOWER": {"REGEX": ".*"}}],
    ],

    "general": [
        [{"LOWER": "new"}],
        [{"LOWER": "bored"}],
        [{"LOWER": "what"}],[{"LOWER": "is"}],[{"LOWER": "your"}],[{"LOWER": "name"}],
    ],    
}


fast_words = ["faster","quicker"]
slow_words = ["slower","calmer"]
loud_words = ["louder","screamer"]
quiet_words = ["softer","quieter"]
#


for category, patterns in responses.items():
    for pattern in patterns:
        matcher.add(category, [pattern])




# In[ ]:


def general(user_input):
    ##fill with general conversation about 
    ##history of prevois questions
    newString = "Hello"
    if user_input.find("name")!=-1:
        newString = "My name is DJ ORPHEUS, no need tell me yours"
    elif user_input.find("bored")!=-1:
        newString = "Thats actually crazy"

    return newString


# In[83]:


def search_spotify(genres, tempo):
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    client_id = 'b0715167b5814e2c92afb73034ed1416'
    client_secret = 'f25f4de9272c49948019dc270b2413d8'

    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



    seed_genres = [genres[0]]
    target_tempo = int(tempo)
    min = target_tempo * 0.9
    max = target_tempo * 1.1


    recommendations = sp.recommendations(seed_genres=seed_genres,  target_tempo=(min, max))

    if recommendations['tracks']:
        first_track = recommendations['tracks'][0]
        track_name = first_track['name']
        artist_name = first_track['artists'][0]['name']

        return f'Song: {track_name} by {artist_name}'
    else:
        return 'No recommendations found from spotify.'


# In[84]:


def chatbot_response(user_input, amoSim, features1=None):
    doc = nlp(user_input)
    matches = matcher(doc)
    if matches:
        match_id, start,end = matches[0]
        category = nlp.vocab.strings[match_id]
        if category == "greetings":
           return "Hello! How can I assist you?"
        elif category == "inquiries":
           return "I'm just the world's best DJ. How can I assist you?"
        elif category == "like":
            print("Loading....")  
            # extracted_word = doc[1].text
            before, keyword,extracted_word = doc.text.partition(doc[1].text)
            search(extracted_word)
            features1 = extract_features(r"music/downloaded/musicaudio.mp3")
            genre1 = xgb.predict(features1)
            genreProb = xgb.predict_proba(features1)
            features1['label'] = genre1[0]
            label = label_encoder.inverse_transform(features1['label'])[0]
            high = confidence_score(genreProb)
            # print("Confidence Scores of song Genre")
            # for i in high:
            #     print(i[0]," :",i[1],"%")

            strLabel = "You ", keyword , extracted_word," They seem to make " ,label," Im saying with", high[0][1],"% confidence"
            return strLabel, high
        elif category == "find_increased":
            if features1 is None:
                return "Exctract a song to use this great feature"
            else:
                words = [token.text for token in doc if token.is_alpha]
                valid = False
                for s in words:
                    if(s in fast_words):
                        features = 'tempo'
                        value = features1[features]
                        value = value/100
                        valid=True
                    elif(s in slow_words):
                        features = 'tempo'
                        value = features1[features]
                        value = -value/100
                        valid=True
                    elif(s in loud_words):
                        features = 'rms_mean'
                        value = 50
                        valid=True
                    elif(s in quiet_words):
                        features = 'rms_mean'
                        value = -50
                        valid=True
                if(valid):
                    new_features = features1
                    new_features[features]+= value
                    print(value)
                    return find_sim(new_features)
                else:
                    return "I'm sorry, but im going to need a valid song feature"
                
        elif category == "find_sim":      
            if features1 is None:
                return "Exctract a song to use this great feature"
            else:
                if amoSim>=3:
                    amoSim=0
                    return "I just put on some back to back bangers!!!"
                else:
                    amoSim = amoSim+1
                    sim = find_sim(features1)
                    for key, value in sim.items():
                        print(key," :",round(value,2),"% similiar")
                    label = label_encoder.inverse_transform(features1['label'])[0]
                    print("Recommendation from Spotify: ",search_spotify(label,features1['tempo']))
                    return "Similiar Songs"
            
        elif category=="general":
            extracted_word = doc.text
            return general(extracted_word)
    else:
        return "I'm sorry, I don't understand that."


# # Main

# * add extract feature depending on download location
# * add more word features in the dictionary
# * clean prints
# * change find_increased features
# * search features change
# * spotify api

# In[85]:


def extract(name):
    print("Loading....")  
    features1 = extract_features(name)
    print("Extracted")
    genre1 = xgb.predict(features1)
    genreProb = xgb.predict_proba(features1)
    features1['label'] = genre1[0]
    label = label_encoder.inverse_transform(features1['label'])[0]
    high = confidence_score(genreProb)
    strLabel="This song is sounding a lot like the ", label," genre. Im saying with ", high[0][1],"% confidence"
    # print("Orpheus: This song is sounding a lot like the", label," genre. Im saying with", high[0][1],"% confidence")
    # print("Confidence Scores")
    # for i in high:
    #     print(i[0]," :",i[1],"%")
    return features1,strLabel, high


# In[86]:

def chat():
    print(f"Orpheus: Hello My Name is DJ ORPHEUS, need some songs im here to help")
    amoSim = 0
    while True:
        user_input = input("You: ")
        if user_input.lower() == "extract":
            name = "music/downloaded/musicaudio.mp3"
            features1 = extract(name)
        elif user_input.lower() == "exit":
            break
        else:
            try:
                features1
            except NameError:
                response = chatbot_response(user_input, amoSim)
            else:
                response = chatbot_response(user_input,amoSim, features1)
            print(f"Orpheus: {response}")



from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

# @app.route('/api/script', methods=['GET'])
# def script():
#     result = "Hello"
#     return jsonify({'result': result})

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/chat', methods=['POST'])
def chatbot():
    amoSim = 0
    data = request.get_json()
    user_input = data.get('user_input')

    if user_input.lower() == "extract":
        name = "music/downloaded/musicaudio.mp3"
        features1, response,high = extract(name)
        return jsonify({"status":"OK","Orpheus": response, "confidence":high})

    else:
        try:
            features1=""
        except NameError:
            response,high = chatbot_response(user_input, amoSim)
            return jsonify({"status":"OK","Orpheus": response, "confidence":high})

        else:
            response = chatbot_response(user_input, amoSim, features1)
            return jsonify({"status":"OK","Orpheus": response})





if __name__ == '__main__':
    app.run(debug=True)

# In[87]:


#cache results 
#https://www.turing.com/kb/a-comprehensive-guide-to-named-entity-recognition

