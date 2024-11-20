import os
import tarfile
import zipfile
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_data_from_tgz(tgz_path):
    extracted_data = []
    if not os.path.exists(tgz_path):
        logging.warning(f"Файл {tgz_path} не существует. Пропускаем.")
        return extracted_data

    try:
        with tarfile.open(tgz_path, 'r:gz') as archive:
            for member in archive.getmembers():
                if 'package' in member.name:
                    file = archive.extractfile(member)
                    if file:
                        file_content = file.read().decode('utf-8', errors='ignore')
                        extracted_data.append({'file_name': member.name, 'content': file_content})
    except Exception:
        logging.warning(f"Не удалось обработать архив {tgz_path}. Пропускаем файл.")
    return extracted_data



def extract_data_from_zip(zip_path): #Извлечение данных из .zip архива
    extracted_data = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for member in archive.namelist():
                if 'package' in member:
                    with archive.open(member) as file:
                        file_content = file.read().decode('utf-8', errors='ignore')
                        extracted_data.append({'file_name': member, 'content': file_content})
    except Exception as e:
        logging.error(f"Ошибка при обработке {zip_path}: {e}")
    return extracted_data


def prepare_features(data): #Извлечение текстовых данных для обработки
    return [file_data['content'] for file_data in data]

#Загрузка архивов из папки с использованием функции-экстрактора
def load_archives(folder_path, extractor_func):
    all_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            all_data.extend(extractor_func(file_path))
    return all_data

#Обучение модели:
def train_model(benign_folder_path, malicious_folder_path):
    logging.info("Загрузка данных...")
    benign_data = load_archives(benign_folder_path, extract_data_from_tgz)
    malicious_data = load_archives(malicious_folder_path, extract_data_from_zip)

    all_data = benign_data + malicious_data
    labels = [0] * len(benign_data) + [1] * len(malicious_data)

    logging.info("Подготовка данных...")
    features = prepare_features(all_data)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.001, random_state=42)

    logging.info("Обучение модели...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Точность модели: {accuracy * 100:.2f}%")

    # Сохранение модели и векторизатора
    joblib.dump(model, 'benign_malicious_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    logging.info("Модель и векторизатор успешно сохранены.")


def classify_file(file_path):
    try:
        model = joblib.load('benign_malicious_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError:
        logging.warning("Модель или векторизатор отсутствуют. Обучите модель перед классификацией.")
        return

    if file_path.endswith('.tgz'):
        extracted_data = extract_data_from_tgz(file_path)
    elif file_path.endswith('.zip'):
        extracted_data = extract_data_from_zip(file_path)
    else:
        logging.warning("Неподдерживаемый формат файла. Ожидается .tgz или .zip.")
        return

    if not extracted_data:
        logging.warning(f"Файл {file_path} пустой или его невозможно обработать.")
        return

    features = prepare_features(extracted_data)
    X_new = vectorizer.transform(features)
    predictions = model.predict(X_new)

    if all(pred == 0 for pred in predictions):
        logging.info(f"Файл {file_path} классифицирован как 'benign' (чистый).")
    else:
        logging.info(f"Файл {file_path} классифицирован как 'malicious' (вредоносный).")

def main():
    benign_folder_path = 'benign'  # Папка с чистыми данными
    malicious_folder_path = 'malware'  # Папка с вредоносными данными

    # Обучение модели
    train_model(benign_folder_path, malicious_folder_path)

    # проверка
    folder_path = "test"
    file_list = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    for _ in file_list:
        test_file_path = f"test/{_}"
        classify_file(test_file_path)


if __name__ == '__main__':
    main()
