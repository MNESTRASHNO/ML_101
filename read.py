import joblib

# Загрузка модели
model = joblib.load('benign_malicious_model.pkl')

# Загрузка векторизатора
vectorizer = joblib.load('vectorizer.pkl')

# Использование загруженных объектов
print(type(model))
print(type(vectorizer))
