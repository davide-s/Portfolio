import tkinter as tk                
from tkinter import font  as tkfont 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import spotipy
import spotipy.util as util
from tkinter import Radiobutton
sp = spotipy.Spotify() 
from spotipy.oauth2 import SpotifyClientCredentials 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import itertools
import threading
import time
import sys


p = None
scoreDecisionTree=None
scoreKnn=None
scoreRandomForest=None
scoreKmeans=None
forest = None
k_means = None
knn = None
c = None
features = None
pred = None
target_playlist = None


done = False
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

t = threading.Thread(target=animate)
t.start()

#long process here
time.sleep(10)
done = True



class SpotifyApp(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(
            family='Helvetica', size=18, weight="bold", slant="italic")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.frames["StartPage"] = StartPage(parent=container, controller=self)
        self.frames["PageOne"] = PageOne(parent=container, controller=self)
        self.frames["PageTwo"] = PageTwo(parent=container, controller=self)

        self.frames["StartPage"].grid(row=0, column=0, sticky="nsew")
        self.frames["PageOne"].grid(row=0, column=0, sticky="nsew")
        self.frames["PageTwo"].grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
    def update_globals(self):
        
        frame = self.frames["PageOne"]
        frame.label1.config(text="Decision tree: {0:.2f} %".format(round(scoreDecisionTree,2)))
        frame.label2.config(text="Knn: {0:.2f} %".format(round(scoreKnn,2)))
        frame.label3.config(text="Random forest: {0:.2f} %".format(round(scoreRandomForest,2)))
        frame.label4.config(text="Kmeans: {0:.2f} %".format(round(scoreKmeans,2)))
        
    def update_globals2(self):
        i=0
        frame = self.frames["PageTwo"]
        if pred is not None:
            for prediction in pred:
                if(prediction == 1):
                    label = tk.Label(frame, text="Titolo:  "+ target_playlist["song_title"][i] + ", Artista: "+ target_playlist["artist"][i]+ ",  Probabilita gradimento: {0:.2f} %".format(round((target_playlist["percentuali"][i]*100),2)), font="Times 8")
                    label.pack(side="top", fill="x")
                i = i +1


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        #sp = login()
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(
            self, text="ID playlist canzoni piaciute", font="Times 15")
        label.pack(side="top", fill="x", pady=10)

        self.entry1 = tk.Entry(self, width=30)
        self.entry1.pack(side="top", fill="x", pady=10)

        label = tk.Label(
            self, text="ID playlist canzoni non piaciute", font="Times 15")
        label.pack(side="top", fill="x", pady=10)

        self.entry2 = tk.Entry(self, width=30)
        self.entry2.pack(side="top", fill="x", pady=10)

        def parametri():
            estrazioneCanzoni(sp, self.entry1.get(), self.entry2.get())
            controller.update_globals()
            controller.show_frame("PageOne")

        button1 = tk.Button(self, text="Analizza", command=lambda: parametri())

        button1.pack()
        


class PageOne(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller

        label0 = tk.Label(self, text="Accuratezza", font="Times 15")
        label0.pack(side="top", fill="x", pady=10)
        label = tk.Label(self, text="L'accuratezza viene calcolata sul test set", font="Times 8")
        label.pack(side="top", fill="x", pady=10)
        self.label1 = tk.Label(self, font="Times 10")
        self.label1.pack(side="top", fill="x", pady=10)
        self.label2 = tk.Label(self, font="Times 10")
        self.label2.pack(side="top", fill="x", pady=10)
        self.label3 = tk.Label(self, font="Times 10")
        self.label3.pack(side="top", fill="x", pady=10)
        self.label4 = tk.Label(self, font="Times 10")
        self.label4.pack(side="top", fill="x", pady=10)
        
        
        label5 = tk.Label(self, text="Seleziona l'algoritmo da utilizzare", font="Times 15")
        label5.pack(side="top", fill="x", pady=10)
        
        var = tk.IntVar()
        R1 = Radiobutton(self, text="Decision Tree", variable=var, value=1)
        R1.pack(side="top", fill="x", pady=10)
        R2 = Radiobutton(self, text="Knn", variable=var, value=2)
        R2.pack(side="top", fill="x", pady=10)
        R3 = Radiobutton(self, text="Random forest", variable=var, value=3)
        R3.pack(side="top", fill="x", pady=10)
        R4 = Radiobutton(self, text="Kmeans", variable=var, value=4)
        R4.pack(side="top", fill="x", pady=10)
        
        label6 = tk.Label(
            self, text="ID playlist in cui ricercare le canzoni", font="Times 15")
        label6.pack(side="top", fill="x", pady=10)

        self.entry1 = tk.Entry(self, width=30)
        self.entry1.pack(side="top", fill="x", pady=10)
        
        def par():
            estraiPreferite(self.entry1.get(), var.get())
            controller.update_globals2()
            controller.show_frame("PageTwo")
            
        
        button = tk.Button(self, text="Cerca",
                           command=lambda: par())
        button.pack()
        
        button1 = tk.Button(self, text="Indietro", command=lambda:  controller.show_frame("StartPage"))

        button1.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
       
        self.label = tk.Label(self, font="Times 10")
        self.label.pack(side="top", fill="x", pady=5)
               
        
        button = tk.Button(self, text="Nuovo",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()
        button1 = tk.Button(self, text="Indietro",
                           command=lambda: controller.show_frame("PageOne"))
        button1.pack()
   
   
        

def login():
    cid ='1672c3b2a8944d548dfb1401fd1149da'  
    secret = 'd875d40e005348999285a962e6f03759' 
    username = '212rqsy4mkk3rfm2nznp6kakq' 

    scope = 'user-library-read playlist-modify-public playlist-read-private'

    redirect_uri='https://www.google.it/' # Paste your Redirect URI here

    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    token = util.prompt_for_user_token(username, scope, cid, secret, redirect_uri)

    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Errore durante l'accesso")
    
    return sp



def estrazioneCanzoni(sp,codice_playlist_piaciute,codice_playlist_non_piaciute):
    
    global scoreDecisionTree, scoreKnn, scoreRandomForest, scoreKmeans
    global forest, k_means, knn, c, features
    
    playlist_piaciute = sp.user_playlist('212rqsy4mkk3rfm2nznp6kakq', codice_playlist_piaciute)
    playlist_non_piaciute = sp.user_playlist('212rqsy4mkk3rfm2nznp6kakq', codice_playlist_non_piaciute)
            
    tracce_piaciute = playlist_piaciute["tracks"]
    canzoni_piaciute = tracce_piaciute["items"] 
    while tracce_piaciute['next']:
        tracce_piaciute = sp.next(tracce_piaciute)
        for item in tracce_piaciute["items"]:
            canzoni_piaciute.append(item)
    good_ids = [] 
    print(len(canzoni_piaciute))
    for i in range(len(canzoni_piaciute)):
        good_ids.append(canzoni_piaciute[i]['track']['id'])

    

    tracce_non_piaciute = playlist_non_piaciute["tracks"]
    canzoni_non_piaciute = tracce_non_piaciute["items"] 
    while tracce_non_piaciute['next']:
        tracce_non_piaciute = sp.next(tracce_non_piaciute)
        for item in tracce_non_piaciute["items"]:
            canzoni_non_piaciute.append(item)
    bad_ids = [] 
    print(len(canzoni_non_piaciute))
    for i in range(len(canzoni_non_piaciute)):
        bad_ids.append(canzoni_non_piaciute[i]['track']['id'])

           
    features = []
    j = 0
    for i in range(0,len(good_ids),50):
        audio_features = sp.audio_features(good_ids[i:i+50])
        for track in audio_features:
            features.append(track)
            track = canzoni_piaciute[j]
            j= j+1
            features[-1]['trackPopularity'] = track['track']['popularity']
            features[-1]['artistPopularity'] = sp.artist(track['track']['artists'][0]['id'])['popularity']
            features[-1]['target'] = 1
    j = 0
    for i in range(0,len(bad_ids),50):
        audio_features = sp.audio_features(bad_ids[i:i+50])
        for track in audio_features:
            features.append(track)
            track = canzoni_piaciute[j]
            j= j+1
            features[-1]['trackPopularity'] = track['track']['popularity']
            features[-1]['artistPopularity'] = sp.artist(track['track']['artists'][0]['id'])['popularity']
            features[-1]['target'] = 0
            
    trainingData = pd.DataFrame(features) 
    print(len(trainingData))
    
    train, test = train_test_split(trainingData, test_size = 0.15)
    print("Dimensione training set: {}, Dimensione test set: {}".format(len(train),len(test)))
    
    
    """"
    
red_blue = ['#19B5FE', '#EF4836']
palette = sns.color_palette(red_blue)
sns.set_palette(palette)
sns.set_style('white')

pos_tempo = trainingData[trainingData['target'] == 1]['tempo']
neg_tempo = trainingData[trainingData['target'] == 0]['tempo']
pos_mode = trainingData[trainingData['target'] == 1]['mode']
neg_mode = trainingData[trainingData['target'] == 0]['mode']
pos_time = trainingData[trainingData['target'] == 1]['time_signature']
neg_time = trainingData[trainingData['target'] == 0]['time_signature']
pos_liveness = trainingData[trainingData['target'] == 1]['liveness']
neg_liveness = trainingData[trainingData['target'] == 0]['liveness']
pos_dance = trainingData[trainingData['target'] == 1]['danceability']
neg_dance = trainingData[trainingData['target'] == 0]['danceability']
pos_duration = trainingData[trainingData['target'] == 1]['duration_ms']
neg_duration = trainingData[trainingData['target'] == 0]['duration_ms']
pos_loudness = trainingData[trainingData['target'] == 1]['loudness']
neg_loudness = trainingData[trainingData['target'] == 0]['loudness']
pos_speechiness = trainingData[trainingData['target'] == 1]['speechiness']
neg_speechiness = trainingData[trainingData['target'] == 0]['speechiness']
pos_valence = trainingData[trainingData['target'] == 1]['valence']
neg_valence = trainingData[trainingData['target'] == 0]['valence']
pos_energy = trainingData[trainingData['target'] == 1]['energy']
neg_energy = trainingData[trainingData['target'] == 0]['energy']
pos_acousticness = trainingData[trainingData['target'] == 1]['acousticness']
neg_acousticness = trainingData[trainingData['target'] == 0]['acousticness']
pos_key = trainingData[trainingData['target'] == 1]['key']
neg_key = trainingData[trainingData['target'] == 0]['key']
pos_instrumentalness = trainingData[trainingData['target'] == 1]['instrumentalness']
neg_instrumentalness = trainingData[trainingData['target'] == 0]['instrumentalness']
pos_popularity = trainingData[trainingData['target'] == 1]['trackPopularity']
neg_popularity = trainingData[trainingData['target'] == 0]['trackPopularity']    
#Tempo
fig = plt.figure(figsize=(12,8))
plt.title("Distribuzione tempo")
plt.xlabel('Tempo', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_tempo.hist(alpha=0.7, bins=30, label='positive')
neg_tempo.hist(alpha=0.7, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Time signature
fig = plt.figure(figsize=(12,8))
plt.title("Distribuzione time_signature")
plt.xlabel('Time_signature', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_time.hist(alpha=0.7, bins=30, label='positive')
neg_time.hist(alpha=0.7, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Liveness
fig = plt.figure(figsize=(12,8))
plt.title("Distribuzione liveness")
plt.xlabel('Liveness', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_liveness.hist(alpha=0.7, bins=30, label='positive')
neg_liveness.hist(alpha=0.7, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Mode
fig2 = plt.figure(figsize=(12,12))
plt.title("Distribuzione mode")
plt.xlabel('Mode', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_mode.hist(alpha=0.5, bins=30, label='positive')
neg_mode.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Valence
fig2 = plt.figure(figsize=(12,12))
plt.title("Distribuzione valence")
plt.xlabel('Valence', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_valence.hist(alpha=0.5, bins=30, label='positive')
neg_valence.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Danceability
fig3 = plt.figure(figsize=(12,12))
plt.title("Distribuzione danceability")
plt.xlabel('Danceability', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_dance.hist(alpha=0.5, bins=30, label='positive')
neg_dance.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Duration
fig3 = plt.figure(figsize=(12,12))
plt.title("Distribuzione duration_ms")
plt.xlabel('Duration_ms', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_duration.hist(alpha=0.5, bins=30, label='positive')
neg_duration.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Loudness
fig3 = plt.figure(figsize=(12,12))
plt.title("Distribuzione loudness")
plt.xlabel('Loudness', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_loudness.hist(alpha=0.5, bins=30, label='positive')
neg_loudness.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Energy
fig3 = plt.figure(figsize=(12,12))
plt.title("Distribuzione energy")
plt.xlabel('Energy', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_energy.hist(alpha=0.5, bins=30, label='positive')
neg_energy.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Key
fig3 = plt.figure(figsize=(12,12))
plt.title("Distribuzione key")
plt.xlabel('Key', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_key.hist(alpha=0.5, bins=30, label='positive')
neg_key.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Speechiness
fig3 = plt.figure(figsize=(12,12))
plt.title("Distribuzione speechiness")
plt.xlabel('Speechness', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_speechiness.hist(alpha=0.5, bins=30, label='positive')
neg_speechiness.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Acousticness
fig3 = plt.figure(figsize=(12,12))
plt.title("Distribuzione acousticness")
plt.xlabel('Acousticness', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_acousticness.hist(alpha=0.5, bins=30, label='positive')
neg_acousticness.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()

#Instrumentalness
fig3 = plt.figure(figsize=(12,12))
plt.title("Distribuzione instrumentalness")
plt.xlabel('Instrumentalness', fontsize=14)
plt.ylabel('Canzoni', fontsize=14)
pos_instrumentalness.hist(alpha=0.5, bins=30, label='positive')
neg_instrumentalness.hist(alpha=0.5, bins=30, label='negative')
plt.legend(loc='upper right')
plt.show()
"""
    
    
    #Define the set of features that we want to look at
    features = ["danceability", "loudness", "valence", "energy", "acousticness","speechiness", "tempo"]
    #Split the data into x and y test and train sets to feed them into a bunch of classifiers!
    x_train = train[features]
    y_train = train["target"]

    x_test = test[features]
    y_test = test["target"]


    c = DecisionTreeClassifier(min_samples_split=100)
    dt = c.fit(x_train, y_train)
    y_pred = c.predict(x_test)
    scoreDecisionTree = accuracy_score(y_test, y_pred) * 100
    print("Accuracy usando Decision Tree: ", round(scoreDecisionTree, 1), "%")

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)
    scoreKnn = accuracy_score(y_test, knn_pred) * 100
    print("Accuracy usando Knn: ", round(scoreKnn, 1), "%")


    forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    forest.fit(x_train, y_train)
    forest_pred = forest.predict(x_test)
    scoreRandomForest = accuracy_score(y_test, forest_pred) * 100
    print("Accuracy usando random forest: ", round(scoreRandomForest, 1), "%")


    k_means = KMeans(n_clusters=3, random_state=0)
    k_means.fit(x_train, y_train)
    predicted= k_means.predict(x_test)
    scoreKmeans = accuracy_score(y_test, predicted)*100
    print("Accuracy usando Kmeans: ", round(scoreKmeans, 1), "%")


    train, test = train_test_split(trainingData, test_size = 0.15)
    print("Dimensione training set: {}, Dimensione test set: {}".format(len(train),len(test)))
    
    
def estraiPreferite(codice_playlist_preferite, algoritmo_scelto):
    
    global pred,target_playlist,p
    
    target_playlist = sp.user_playlist("spotify", codice_playlist_preferite)

    newPlaylist_tracce = target_playlist["tracks"]
    newPlaylist_canzoni = newPlaylist_tracce["items"] 
    while newPlaylist_tracce['next']:
        newPlaylist_tracce = sp.next(newPlaylist_tracce)
        for song in newPlaylist_tracce["items"]:
            newPlaylist_canzoni.append(song)
        
    newPlaylist_song_ids = [] 
    print(len(newPlaylist_canzoni))
    for i in range(len(newPlaylist_canzoni)):
        newPlaylist_song_ids.append(newPlaylist_canzoni[i]['track']['id'])
    
    newPlaylist_features = []
    j = 0
    for i in range(0,len(newPlaylist_song_ids),50):
        audio_features = sp.audio_features(newPlaylist_song_ids[i:i+50])
        for track in audio_features:
            track['song_title'] = newPlaylist_canzoni[j]['track']['name']
            track['artist'] = newPlaylist_canzoni[j]['track']['artists'][0]['name']
            j= j + 1
            newPlaylist_features.append(track)

    target_playlist = pd.DataFrame(newPlaylist_features)

    if(algoritmo_scelto==1):
        pred = c.predict(target_playlist[features])
        p = c.predict_proba(target_playlist[features])
    if(algoritmo_scelto==2):
        pred = knn.predict(target_playlist[features])
        p = knn.predict_proba(target_playlist[features])
    if(algoritmo_scelto==3):
        pred = forest.predict(target_playlist[features])
        p = forest.predict_proba(target_playlist[features])
    if(algoritmo_scelto==4):
        pred = k_means.predict(target_playlist[features])
        p = k_means.predict_proba(target_playlist[features])

    likedSongs = 0
    i = 0
    print("pre sorting")
    for i,prediction in enumerate(pred):
        target_playlist['percentuali'] = pd.Series(item[1] for item in p)
       
        
    print(target_playlist['percentuali'])
           
    target_playlist.sort_values(by=['percentuali'],inplace=True, ascending=False)
    target_playlist.reset_index(inplace=True)
    print("post sorting")
    
    print(target_playlist['percentuali'])
    
    
    i=0
    for i in enumerate(pred):
            print ("Titolo: " + target_playlist["song_title"][i] + ",  Artista:  "+ target_playlist["artist"][i] + ",  Probabilita gradimento: {0:.2f} %".format(round((target_playlist["percentuali"][i]*100),2))  )
            likedSongs= likedSongs + 1
    print(likedSongs);
    


if __name__ == "__main__":
    sp = login()
    app = SpotifyApp()
    app.mainloop()
