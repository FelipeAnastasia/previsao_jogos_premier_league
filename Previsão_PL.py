##Felipe Anastasia
##Previsão da PL usando scikit-learn para prever da base de dados matches.csv que contém todos os jogos no período de 2022-2020

import pandas as pd
from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)

matches = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/ProjetosCDIA/matches.csv', index_col = 0)
matches

##converter todos os objetos para int ou float para serem processados pelo ML
matches["date"] = pd.to_datetime(matches["date"])
matches["h/a"] = matches["venue"].astype("category").cat.codes ##convertendo localização casa (1) ou fora (0) para um número
matches["opp"] = matches["opponent"].astype("category").cat.codes ##convertendo oponentes para um número
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int") ##convertendo horas para número caso algum time jogue melhor em algum horário
matches["day"] = matches["date"].dt.dayofweek ##convertendo dia da semana do jogo para um número

matches["target"] = (matches["result"] == "W").astype("int") ##atribuindo um valor para vitória (1)

from sklearn.ensemble import RandomForestClassifier ##importando ML para data não linear

rf = RandomForestClassifier(n_estimators = 100, min_samples_split=10, random_state=1)
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
predictors = ["h/a", "opp", "hour", "day"]
rf.fit(train[predictors], train["target"])
RandomForestClassifier(min_samples_split = 10, n_estimators = 100, random_state = 1)
preds = rf.predict(test[predictors]) ##treinando a máquina e fazendo a previsão

from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"], preds) ##testando a precisão
acc
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
pd.crosstab(index=combined["actual"], columns=combined["prediction"])

from sklearn.metrics import precision_score
precision_score(test["target"], preds)

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester United").sort_values("date")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date") ##organizando jogos por data
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) ##descartando valores faltantes e substituindo por vazio
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols] ##criando novas colunas com os valores da função rolling averages

rolling_averages(group, cols, new_cols) ##chamando a função e gerando a média dos últimos 3 jogos

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team') ##descartando o index extra

matches_rolling.index = range(matches_rolling.shape[0]) ##adicionando novo index
matches_rolling

def make_predictions(data, predictors): ##fazendo as previsões
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors]) ##fazendo a previsão
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision ##retornando os valores para a previsão

combined, precision = make_predictions(matches_rolling, predictors + new_cols)

precision

combined

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index = True, right_index = True)
combined

class MissingDict(dict): ##criando uma classe que herda da classe dicionário
    __missing__ = lambda self, key: key ##caso o nome de algum time não esteja aparecendo

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)
mapping["West Ham United"]

combined["new_team"] = combined["team"].map(mapping)
combined

merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"]) ##juntando a previsão do time de casa com o time de fora
merged

##projeto inspirado pelo tutorial do dataquest
