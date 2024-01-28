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
from pytube import YouTube
from pydub import AudioSegment
import youtube_dl
import os
import xgboost as xgbo
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import xgboost
import joblib


#model
df = pd.read_csv(r'music_data/features_3_sec.csv')
df = df[['chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','tempo','label']]
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['label'] =  label_encoder.fit_transform(df['label'])
# print(label_encoder.classes_)
# y = df[['label']]
# X = df[df.columns.difference(['label'])]

## split both X and y using a ratio of 70% training - 30% testing
##add min maxing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# print(len(X_train), len(X_test), len(y_train), len(y_test))

# xgb = xgboost.XGBClassifier(n_estimators=1000,enable_catergorical=True,learning_rate=0.05)
# xgb.fit(X_train, y_train)

# predictions = xgb.predict(X_test)

target_name = ['blues', 'classical', 'country', 'disco', 'hiphop' ,'jazz' ,'metal', 'pop','reggae' ,'rock']

# print(classification_report(y_test, predictions, target_names=target_name))
# print("Accuracy: " ,metrics.accuracy_score(y_test, predictions))
# cols_when_model_builds = xgb.feature_names

xgb = joblib.load('model.pkl')
cols_when_model_builds = xgb.get_booster().feature_names




#find sim music function
def find_sim(data):
    placeHoldername = 'test'
    data['filename'] = placeHoldername

    df_sim = pd.read_csv(r'music_data/features_30_sec.csv')

    df_sim = df_sim[['filename','chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','tempo','label']]

    df_sim['label'] = df_sim['label'].astype("string")
    df_sim['label'] =  label_encoder.fit_transform(df_sim['label'])


    combined_df = pd.concat([df_sim, data], ignore_index=True)

    combined_df = combined_df.set_index('filename')

    genre = combined_df[['label']]

    #https://naomy-gomes.medium.com/the-cosine-similarity-and-its-use-in-recommendation-systems-cb2ebd811ce1 + https://www.kaggle.com/code/andradaolteanu/work-w-audio-data-visualise-classify-recommend/notebook#Machine-Learning-Classification
    scaled = preprocessing.scale(combined_df)
    cos_similarity = cosine_similarity(scaled)
    new_data = pd.DataFrame(cos_similarity)
    new_data_names = new_data.set_index(genre.index)
    new_data_names.columns = genre.index

    series = new_data_names[placeHoldername].sort_values(ascending=False)
    series = series.drop(placeHoldername)

    series = series.head(3).to_dict()

    from collections import Counter

    k = Counter(series)
    
    # Finding 3 highest values
    series = k.most_common(3) 
    
    for i in series:
        print(i[0]," :",i[1]," ")
    return series



#users linear regression to predict features of previusly liked music
def find_pred(result):
    import json
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()

    dfs = []

    if(result is None):
        return None
    

    df = pd.DataFrame.from_dict(result, orient='index')
    dfs.append(df)


    combined_df = pd.concat(dfs, ignore_index=True)

    newData=[]
    for i in range(len(df)):
        y = json.loads(combined_df[0][i])
        newData.append(y)

    if(len(newData)<3):
        return None

    newData = pd.DataFrame(newData)

    finalDf = pd.DataFrame()
    for i in range(len(newData)):
        df = pd.DataFrame([newData[0][i]])
        finalDf = pd.concat([finalDf, df], ignore_index=True)


    # finalDf =  finalDf.drop(['label'], axis=1)

    features = pd.DataFrame(columns=finalDf.columns)

    for column in finalDf:
        X_train, X_test, y_train, y_test = train_test_split(finalDf, finalDf[column], test_size=0.2)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        features.loc[0, column] = prediction[0]  # Assuming you want to update the first row (index 0)
        # print(column,":",prediction[0])
        
    



    return features




#returns the highest scores
def confidence_score(proba):
    from collections import Counter
    confi = {}
    i=0
    for val in proba[0]:
        rounded = round(val *100,2)
        confi[target_name[i]] = rounded
        i = i+1

    counter = Counter(confi)
    
    high = counter.most_common(3) 
    return high


#extract mfeatures from a song
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
    print(features)


    return features

#search youtube for a song
def search(query):
    from pytube import YouTube
    from googleapiclient.discovery import build

    api_key = 'AIzaSyCghPkifWFcLs_iN5CCvLIlQwvWBXxIxxY'

    youtube = build('youtube', 'v3', developerKey=api_key)

    search_response = youtube.search().list(q=query,type='video',part='id,snippet',maxResults=1).execute()

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



import spacy
from spacy.matcher import Matcher
spacy.cli.download("en_core_web_sm")

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
    "predicitions": [
        [{"LOWER": "recos"}],
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


def general(user_input):
    ##fill with general conversation about 
    ##history of prevois questions
    newString = "Hello"
    if user_input.find("name")!=-1:
        newString = "My name is DJ ORPHEUS, no need tell me yours"
    elif user_input.find("bored")!=-1:
        newString = "Thats actually crazy"

    return newString


def search_spotify(genres, tempo):
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    SPOTIPY_CLIENT_ID = 'b0715167b5814e2c92afb73034ed1416'
    SPOTIPY_CLIENT_SECRET = 'f25f4de9272c49948019dc270b2413d8'

    client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
    spotifySearcher = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



    seed_genres = [genres[0]]
    target_tempo = int(tempo)
    min = target_tempo * 0.9
    max = target_tempo * 1.1


    recommendations = spotifySearcher.recommendations(seed_genres=seed_genres,  target_tempo=(min, max))

    if recommendations['tracks']:
        song = recommendations['tracks'][0]
        song_name = song['name']
        artist_name = song['artists'][0]['name']

        return f'Song: {song_name} by {artist_name}'
    else:
        return 'No recommendations found from spotify.'


def chatbot_response(user_input, amoSim, features1=None, userID=None):
    doc = nlp(user_input)
    matches = matcher(doc)
    if matches:
        match_id, start,end = matches[0]
        category = nlp.vocab.strings[match_id]
        if category == "greetings":
            return "Hello! How can I assist you?",None,None,None,None
        elif category == "inquiries":
           return "I'm just the world's best DJ. How can I assist you?",None,None,None,None
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

            strLabel = "You ", keyword , extracted_word," They seem to make " ,label," Im saying with", high[0][1],"% confidence"
            return strLabel, None, features1,None, high
        elif category == "find_increased":
            if features1 is None:
                strLabel=  "Exctract a song to use this great feature"
                
                return strLabel,None,None,None,None
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
                    songs = find_sim(new_features)
                    return "Here are some of those increased features", songs, features1,None,None
                else:
                    strLabel = "I'm sorry, but im going to need a valid song feature"
                    return strLabel, None,None,None,None
                
        elif category == "find_sim":      
            if features1 is None:
                strLabel=  "Exctract a song to use this great feature"
                return strLabel,None,None,None,None           
            else:
                if amoSim>=3:
                    amoSim=0
                    return "I just put on some back to back bangers!!!"
                else:
                    amoSim = amoSim+1
                    sim = find_sim(features1)
                    songs=[]
                    for key, value in sim.items():
                        print(key," :",round(value,2),"% similiar")

                    label = label_encoder.inverse_transform(features1['label'])[0]
                    spotifySong = "Recommendation from Spotify: ",search_spotify(label,features1['tempo'])
                    
                    return "Similiar Songs", sim,features1, spotifySong,None
            
        elif category=="general":
            extracted_word = doc.text
            return general(extracted_word),None,None,None,None
        
        elif category=="predicitions":
            print("predicitions")
            from firebase import firebase
            firebase = firebase.FirebaseApplication('https://orpheus-3a4fa-default-rtdb.europe-west1.firebasedatabase.app/', None)
            result = firebase.get('/users', userID)
            pred= find_pred(result)

            if pred is None:
                return "Upload some more songs", None,None, None,None

            sim = find_sim(pred)
            songs=[]
            # for key, value in sim.items():
            #     print(key," :",round(value,2),"% similiar")

            # label = label_encoder.inverse_transform(pred['label'])[0]
            # spotifySong = "Recommendation from Spotify: ",search_spotify(label,pred['tempo'])
                    
            return "I have some songs that i think you might like", sim,pred, None,None
            
    else:
        return "I'm sorry, I don't understand that.",None,None,None,None


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
    return features1,strLabel, high


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

@app.route('/upload', methods=['POST'])
def upload():
    print("upload: ", request.form.get('user_input'))
    data = request.files['music_file']

    data.save('downloadedTest.mp3')
    print(data) 
    return jsonify({"status":"OK"})


@app.route('/chat', methods=['POST'])
def chatbot():
    amoSim = 0
    if(request.form.get('user_input') is None):
        data = request.get_json()
        user_input = data.get('user_input')
        extraction=""
    else:
        extraction = request.form.get('user_input')

    if(extraction == 'extract'):
        data = request.files['music_file']
        name = "downloadedTest.mp3"
        data.save(name)
        features, response,high = extract(name)
        features = features.to_json(orient='records')

        return jsonify({"status":"OK","Orpheus": response,"features":features, "confidence":high})
    else:
        if(data.get('features')!=None):
            features1 = data.get('features')
            features1 = pd.read_json(features1)
        else:
            features1=None

        userID = data.get('userID')
        print("User: ",userID)
        response,songs,features,recommendation, high = chatbot_response(user_input, amoSim, features1, userID=userID)

        if isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
            if(features.empty != True):
                features = features.to_json(orient='records')

        if isinstance(songs, pd.DataFrame) or isinstance(songs, pd.Series):
            if(songs.empty != True):
                songs = songs.to_json()


        return jsonify({"status":"OK","Orpheus": response,"songs":songs, "features": features,"recommendation": recommendation,"confidence":high })

if __name__ == '__main__':
    app.run(debug=True)