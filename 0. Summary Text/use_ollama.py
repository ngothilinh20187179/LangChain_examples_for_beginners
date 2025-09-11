from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama

def main():
    information = """
    Nguyễn Thị Bình; born Nguyễn Thị Châu Sa, also known as Madame Bình,is a South Vietnamese revolutionary leader, diplomat and politician. She became internationally known for her role as the Vietnamese's chief diplomat and leading its delegation to the Paris Peace Conference.
    The only woman among the signatories of the 1973 peace accords that ended American in the War in Vietnam, she later served in the government of reunified Vietnam after that and later became the country's Vice President in 1992. She is the first woman in Vietnamese history to be appointed a cabinet minister.
    Nguyễn Thị Bình was born in 1927 in Châu Thành, Sa Đéc Province and is a granddaughter of the Nationalist leader Phan Chu Trinh. She studied French at Lycée Sisowath in Cambodia and worked as a teacher during the French colonisation of Vietnam. She joined Vietnam's Communist Party in 1948. 
    From 1945 to 1951, she took part in various intellectual movements against the French colonists. Subsequently, she was arrested and jailed between 1951 and 1953 in Chí Hòa Prison (Saigon) by the French colonial authority in Vietnam.
    In 1969 she was appointed foreign minister of the Provisional Revolutionary Government of the Republic of South Vietnam. 
    A fluent French speaker, Bình played a major role in the Paris Peace Accords, an agreement that was supposed to end the war and restore peace in Vietnam, which entered into force on 17 January 1973.
    """
    summary_template = """
    Based on the information: {information}, I want to:
    - create a short summary about that information, it's about 1 sentence
    - give 1 interesting fact about that information, it's aboit 1 sentence
    """  
    summary_prompt_template=ChatPromptTemplate.from_template(summary_template)  
    chat_ollama = ChatOllama(
        model="gemma3:1b",
        temperature=0
    )
    chain = LLMChain(llm=chat_ollama, prompt=summary_prompt_template)
    response = chain.invoke({"information": information})
    
    print("\n--- Summary Result ---")
    print(response['text'])

if __name__ == "__main__":
    main()
