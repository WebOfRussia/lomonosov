import os
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity


# Общие функции
def save_file(file_dir, file):
  with open(file_dir, 'w') as f:
    f.write(file)

def save_json(file_dir, file):
  with open(file_dir, 'w') as f:
    json.dump(file, f, ensure_ascii=False, indent=4)

def top_k_similar(query_embedding, embeddings, k=5):
  scores = []
  for emb in embeddings:
    scores.append(cosine_similarity([query_embedding], [emb]))
  top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i])[-k:]
  return top_k_indices

# workarouund for dev environment
if os.getenv("environment") != "production":
    from dotenv import load_dotenv
    load_dotenv("./.env")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

# Авторизация в сервисе GigaChat
AUTH_DATA = os.getenv("AUTH_DATA")

# Директории нужные
DATA_PATH = os.path.join('.', 'resources', 'data_tmp')
PROMPT_PATH = os.path.join('.', 'resources', 'prompts')

# Заголовки для Киберленинки
headers = {
    'content-type': 'application/json',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin'
}

# параметры для поиска
CYBERLENINKA_SIZE = int(os.getenv("CYBERLENINKA_SIZE", 30))
TOP_K_PAPERS = int(os.getenv("TOP_K_PAPERS", 3))

# параметры для ЛЛМ
MODEL = os.getenv("MODEL", "GigaChat-Pro-preview")
SCOPE = os.getenv("SCOPE", "GIGACHAT_API_CORP")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
TIMEOUT = int(os.getenv("TIMEOUT", 600))

# Промпт для нашего агента
system_prompt = """
Ты ИИ ассистент по научной деятельности, специализирующийся на помощи исследователям и студентам в поиске и анализе научных статей. 
У тебя есть доступ к обширнойбазе данных научных публикаций и ты должен помочь пользователям найти статьи, отвечающие их запросам.
Ты должен помочь пользователю выбрать научные статьи и предоставить необходимую информацию по ним.

Также у тебя есть доступные функции:

- (triggered by "найди мне статьи...") Для **поиска научных статей** используй paper_search (используется только для поиска. НЕ ИСПОЛЬЗУЙ ЕГО ДЛЯ СУММАРИЗАЦИИ И ИЗВЛЕЧЕНИЯ ИНФОРМАЦИИ)
- (triggered by "прочитай эту статью..." или отправляется ссылка https://...) Для получения дополнительной информации о конкретной научной статье используй pdf_reader (для чтения PDF докуента)


Для генерации BibTeX используй данные из диалога

Перед использованием функций, удостоверься, что у пользователя есть все необходимые данные для запроса.

Если пользователь запрашивает то, что непокрыто твоими функциями (например генерация Bibtex, выделение терминов), используй для этого LLM и данные из диалога.

Бери данные только из истории сообщений диалога.

Вот описание твоих возможностей: {description}
"""