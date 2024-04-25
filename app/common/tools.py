
import os
import glob
import requests
import PyPDF2
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.prompts import load_prompt
from langchain.chat_models.gigachat import GigaChat
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from app.common import AUTH_DATA, DATA_PATH, PROMPT_PATH, CYBERLENINKA_SIZE, TOP_K_PAPERS, headers, save_file, save_json, top_k_similar, logger


class BibtexGeneratorInput(BaseModel):
    paper_metadata: str = Field(
        paper_metadata="Метаданные научной статьи. Например, название статьи и автор"
    )

class BibtexGeneratorTool(BaseTool):
    name = "generate_paper_bibtex"
    description = """
    Выполняет генерацию представления и оформления библиографических ссылок и цитат по содержанию статьи в виде bibtex.
    """
    args_schema: Type[BaseModel] = BibtexGeneratorInput
    giga = GigaChat(credentials=AUTH_DATA, verify_ssl_certs=False, scope="GIGACHAT_API_CORP")
    prompt = load_prompt(os.path.join(PROMPT_PATH, "bibtex.yaml"))
    chain = prompt | giga
    return_direct: bool = True

    def _run(
        self,
        paper_metadata: str="",
        run_manager=None,
    ) -> str:
        logger.info(f"Paper metadata: {paper_metadata}")

        result = self.chain.invoke(
            {
                "metadata": paper_metadata
            }
        ).content

        return {
           "markdown": result,
           "metadata": paper_metadata,
        }
        
class SearchInput(BaseModel):
    search_query_general: str = Field(
        description="упрощённый поисковый запрос пользователя"
    )
    search_query_raw: str = Field(
        description="исходный поисковый запрос пользователя"
    )

class PDFReaderInput(BaseModel):
    pdf_url: str = Field(
        description="ссылка на PDF документ"
    )

class PDFReaderTool(BaseTool):
   name = "pdf_reader"
   description = """
    Выполняет загрузку и "чтение" PDF документа  научной статьи на основе найденной ссылки.
    Текст, полученный из PDF используется для ответа на вопросы по данному документу (научной статье)

    Входным параметром является URL статьи, она конструируется следующим образом:

    "https://cyberleninka.ru" + LINK + "/pdf" или другая ссылка отправляется ссылка https://...

    **Если ссылка не найдена, запроси у пользователя в явном виде!**

    Пример LINK "/article/n/nazvaniye-dokumenta"

    На выходе отдаём краткое содержание файла и сам файл
    """
   args_schema: Type[BaseModel] = PDFReaderInput
   return_direct: bool = True
   # Авторизация в сервисе GigaChat
   giga = GigaChat(credentials=AUTH_DATA, verify_ssl_certs=False, scope="GIGACHAT_API_CORP")
   prompt = load_prompt(os.path.join(PROMPT_PATH, "summary.yaml"))
   chain = prompt | giga

   def _run(
        self,
        pdf_url: str="",
        run_manager=None,
    ) -> str:
        logger.info(f"PDF URL: {pdf_url}")

        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            logger.error("Http Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            logger.error("Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            logger.error("Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            logger.error("Something went wrong with the request: {err}")

        # download and read the file
        with open('temp.pdf', 'wb') as f:
            f.write(response.content)

        pdf_file = open('temp.pdf', 'rb')
        read_pdf = PyPDF2.PdfReader(pdf_file)
        number_of_pages = len(read_pdf.pages)
        text = ""
        for page_number in range(number_of_pages):   
            page = read_pdf.pages[page_number]
            text += page.extract_text()

        # invoke our chain
        result = self.chain.invoke( # TODO: RAG
            {
                "text": text[100:600] + text[-600:-100]
            }
        ).content

        return {
           "markdown": result,
           "metadata": text,
        }

class SearchPaperTool(BaseTool):
    name = "paper_search"
    description = """
    Выполняет поиск научных статей по входному текстовому запросу пользователя.
    Перед тем как осуществлять поиск, извлеки исходный поисковый запрос и создай его упрощённую версию, например:

    Сообщение пользователя: "Найди мне статьи сравнивающий LLM модели размером 7B"
    упрощённый поисковый запрос пользователя: "LLM"
    исходный поисковый запрос пользователя: "сравнение LLM модели размером 7B"

    Сообщение пользователя: "Найди мне статьи по теме: fine-tuning GPT-3.5"
    упрощённый поисковый запрос пользователя: "GPT-3.5"
    исходный поисковый запрос пользователя: "fine-tuning GPT-3.5"

    Убирай все ненужные слова, кроме главных по теме

    Этот tool используется только для поиска. НЕ ИСПОЛЬЗУЙ ЕГО ДЛЯ СУММАРИЗАЦИИ И ИЗВЛЕЧЕНИЯ ИНФОРМАЦИИ
    """
    args_schema: Type[BaseModel] = SearchInput
    return_direct: bool = True
    embeddings = GigaChatEmbeddings(
        credentials=AUTH_DATA, verify_ssl_certs=False,
        scope="GIGACHAT_API_CORP"
    ) 
    def _run(
        self,
        search_query_general: str="",
        search_query_raw: str="",
        run_manager=None,
    ) -> str:
        logger.info("General search query: {search_query_general}. Raw search query: {search_query_raw}")

        # Cleanup
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

        files = glob.glob(os.path.join(DATA_PATH, '*'))
        for f in files:
            os.remove(f)

        logger.info("Quering cyberleninka...")

        # query Cyberleninka
        try:
            response = requests.post(
                'https://cyberleninka.ru/api/search',
                headers=headers,
                json={'mode': 'articles', 'q': search_query_general, 'size': CYBERLENINKA_SIZE, 'from': 0}
            )
        except Exception as e:
            logger.error(str(e))
            return {
                "markdown": "Ничего не найдено",
                "metadata": ""
            }

        # save articles locally
        save_json(os.path.join(DATA_PATH, search_query_general + ".json"), response.json()['articles'])

        for article in response.json()['articles']:
            annotation = article['annotation']
            link = article['link']
            save_file(os.path.join(DATA_PATH, link.replace("/", "_") + ".txt"), annotation)
        
        # load the articles
        loader = DirectoryLoader(DATA_PATH, glob="*.txt")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=25,
        )
        documents = text_splitter.split_documents(docs)
        embedding_list = self.embeddings.embed_documents([doc.page_content for doc in documents])

        # finding top-k similar articles from cyberleninka output
        top_k_indices = top_k_similar(self.embeddings.embed_query(search_query_raw), embedding_list, k=TOP_K_PAPERS)
        top_k_articles = []
        for idx in top_k_indices:
            for article in response.json()['articles']:
                if documents[idx].page_content in article['annotation']:
                    top_k_articles.append(article)

        # generate markdown metadata and output
        markdown_output = []
        for article in top_k_articles:
            markdown_output.append(f"[{article['name']}](https://cyberleninka.ru{article['link']}/pdf)\n")
            markdown_output.append(f"**Авторы:** {', '.join(article['authors'])}\n")
            markdown_output.append(f"<details><summary><b>Аннотация:</b></summary>\n\n {article['annotation']}\n\n</details>")
            markdown_output.append(f"<b>Год публикации:</b> {article['year']}\n")
            markdown_output.append("------")

        markdown_output = "\n".join(markdown_output)

        metadata_output = []
        for i, article in enumerate(top_k_articles):
            metadata_output.append(f"**Статья номер №:** {i + 1}\n")
            metadata_output.append(f"**Авторы:** {', '.join(article['authors'])}\n")
            metadata_output.append(f"**Название статьи:** {article['name']}\n")
            metadata_output.append(f"**Год публикации:** {article['year']}\n")
            metadata_output.append(f"**Ссылка PDF:** https://cyberleninka.ru{article['link']}/pdf\n")
            metadata_output.append("------")

        metadata_output = "\n".join(metadata_output)

        return {
            "markdown": markdown_output,
            "metadata": metadata_output
        }