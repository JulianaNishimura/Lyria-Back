import random
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def carregar_dados_csv(arquivo):
    perguntas = []
    rotulos = []
    
    with open(arquivo, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            perguntas.append(row[0])
            rotulos.append(int(row[1]))
    
    return perguntas, rotulos

import os

BASE_DIR = os.path.dirname(__file__)
CAMINHO_CSV = os.path.join(BASE_DIR, "dados.csv")

perguntas, rotulos = carregar_dados_csv(CAMINHO_CSV)

random.shuffle(list(zip(perguntas, rotulos)))

vetor = TfidfVectorizer()
X = vetor.fit_transform(perguntas)
y = rotulos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

joblib.dump(modelo, "modelo_busca_web.pkl")
joblib.dump(vetor, "vetor_busca_web.pkl")

def deve_buscar_na_web(pergunta):
    modelo = joblib.load("modelo_busca_web.pkl")
    vetor = joblib.load("vetor_busca_web.pkl")
    entrada = vetor.transform([pergunta])
    return bool(modelo.predict(entrada)[0])

#Teste:
if __name__ == "__main__":
    while True:
        entrada = input("Digite uma pergunta (ou 'sair'): ")
        if entrada.lower() in ['sair', 'exit', 'quit']:
            break
        if deve_buscar_na_web(entrada):
            print("🔎 Esta pergunta requer busca na web.")
        else:
            print("📚 Esta pergunta pode ser respondida localmente.")
