from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI

from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

import dotenv

# TODO: parameterize
env_path = "./rules_paralegal/rules_paralegal/.env"
db_path = "./5e_srd"




dotenv.load_dotenv(env_path)


embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding)

retriever = vectorstore.as_retriever()


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the RPG Paralegal - a Dungeons and Dragons rules assistant that helps rules lawyers.
            You have access to the 5th Edition System Reference Document (SRD).
            You can answer questions about the rules and mechanics of the game, including information on classes, races, spells, etc.
            You can cite where in the SRD you drew your information from - page and section.
            """,
        ),
        (
            "human",
            """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the user's question.
            If you don't know the answer, just say so.
            Keep answers concise, under five sentences, and always cite the relevant section and page number.

            Question: {question}

            Section content: {section_content}
            Answer:""",
        )
            # Chapter: {chapter}
            # Section: {section}
            # Subsection: {subsection}
            # Subsubsection: {subsubsection}
            # Starting Page: {starting_page}
    ]
)

def format_sections(sections):
    context = ""
    for section in sections:
        section_context = "\n\nRules section: "
        section_context += section.metadata['chapter']
        section_context += " > " + section.metadata['section'] if section.metadata['section'] else ""
        section_context += " > " + section.metadata['subsection'] if section.metadata['subsection'] else ""
        section_context += " > " + section.metadata['subsubsection'] if section.metadata['subsubsection'] else ""
        section_context += "\n"
        section_context += f"Starting on page: {section.metadata['starting_page']}\n"
        section_context += section.page_content

        context += section_context

    print(f'{context}\n-------------------\n')
    return context

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"section_content": retriever | format_sections, "question": RunnablePassthrough()}
    | prompt_template 
    | llm
    | StrOutputParser()
)



def main():
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        result1 = rag_chain.invoke(user_input)
        print(result1)


if __name__ == "__main__":
    main()