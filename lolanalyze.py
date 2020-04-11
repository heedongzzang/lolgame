import requests
import pandas as pd
import time

sohwan = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/" +'hide on bush' +'?api_key=' + api_key
r = requests.get(sohwan)
r.json()['id'] #소환사의 고유 id

api_key = 'RGAPI-d8c5acbe-1643-44cd-88f3-b7bc1209191d'


challenger = 'https://kr.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5?api_key=' + api_key
r = requests.get(challenger)#그마데이터 호출
league_df = pd.DataFrame(r.json())

league_df.reset_index(inplace=True)#수집한 그마데이터 index정리
league_entries_df = pd.DataFrame(dict(league_df['entries'])).T #dict구조로 되어 있는 entries컬럼 풀어주기
league_df = pd.concat([league_df, league_entries_df], axis=1) #열끼리 결합

league_df = league_df.drop(['index', 'queue', 'name', 'leagueId', 'entries', 'rank'], axis=1)
league_df.to_csv('c:/data/챌린저데이터.csv',index=False,encoding = 'cp949')#중간저장

league_df.info()


league_df['summonerId']


account_id = []
for i in range(len(league_df)):
    try:
        sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key 
        r = requests.get(sohwan)
        time.sleep(1.5)
        while r.status_code == 429:
            sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + api_key 
            r = requests.get(sohwan)
            print(i)
            
        account_id.append(r.json()['accountId'])
        print(i)
    except:
        pass
    
    

match_info_df = pd.DataFrame()
season = str(13)
for i in range(len(account_id)):
    try:
        match0 = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + account_id[i]  +'?season=' + season + '&api_key=' + api_key
        r = requests.get(match0)
        
        while r.status_code == 429:
            time.sleep(5)
            match0 = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + account_id[i] +'?season=' + season + '&api_key=' + api_key
            r = requests.get(match0)
            
        match_info_df = pd.concat([match_info_df, pd.DataFrame(r.json()['matches'])])
    
    except:
        print(i)
len(match_info_df['gameId'].unique())
game_id = match_info_df.drop_duplicates(['gameId'])
match_info_df.columns
game_id.to_csv('c:/data/챌린저game_id.csv',index=False,encoding = 'cp949')#중간저장

api_key='RGAPI-2fd6acf7-bd31-49a9-beba-0bee23583691'
game_id = pd.read_csv('c:/data/챌린저game_id.csv') #중복 제거한 게임id
match_fin = pd.DataFrame()
team_df = pd.DataFrame()
a=0
for i in game_id['gameId']:
    
    match = "https://kr.api.riotgames.com/lol/match/v4/matches/{}?api_key=".format(i) + api_key
    r1 = requests.get(match)
    data = r1.json()
    team_data = pd.DataFrame(data['teams'][0])
    mat = pd.DataFrame(list(r1.json().values()), index=list(r1.json().keys())).T
    match_fin = pd.concat([match_fin,mat])
    team_df=pd.concat([team_df,team_data])
    time.sleep(1.5)
    if a ==500:
        break
    a+=1
    print(a)
    
match_fin['teams']
    
team_df
team_df=team_df.drop(['bans'],axis=1)
team_df=team_df.drop([1,2,3,4])


team_df=team_df.reset_index(drop=True)


a=0
for i in team_df['win']:
    if i=='Win':
        team_df['win'][a]=0
    else:
        team_df['win'][a]=1
    a+=1
    
    
from sklearn.preprocessing import LabelEncoder

for i in range(2,8):
    le = LabelEncoder()
    y = list(team_df.iloc[:,i])
    
    le.fit(y)
    y2 = le.transform(y) 
    
    team_df.iloc[:,i] = y2 



dfwin=pd.Series()
dfwin=team_df['win']

team_df=team_df.drop(['win'],axis=1)
dfwin=dfwin.astype('int64')




import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


x_train, x_test, y_train ,a= train_test_split(team_df, dfwin, test_size=0.3)
x_train, x_test, y_train ,y_test = train_test_split(team_df, dfwin ,test_size=0.3)


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)

y_pred = forest.predict(x_test)

print(list(y_pred))
print(list(y_test))
cp= pd.DataFrame(y_pred,columns=['y_pred'])

cp['y_text']=y_test.reset_index(drop=True)
print(metrics.accuracy_score(y_test, y_pred))



#랜덤포레스트 요소 중요
import matplotlib.pyplot as plt
import numpy as np
type(y_test)
def plot(model):
    n_features = team_df.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), team_df.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
plt.show()

plot(forest)
