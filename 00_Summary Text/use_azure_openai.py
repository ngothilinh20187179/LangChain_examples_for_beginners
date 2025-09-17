from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain

def main():
    load_dotenv()
    
    information = """
    Hồ Chí Minh (born Nguyễn Sinh Cung; 19 May 1890 – 2 September 1969), colloquially known as Uncle Ho (Bác Hồ) among other aliases and sobriquets, was a Vietnamese revolutionary and politician who served as the founder and first president of the Democratic Republic of Vietnam from 1945 until his death in 1969, and as its first prime minister from 1945 to 1955. Ideologically a Marxist–Leninist, he founded the Indochinese Communist Party in 1930 and its successor Workers' Party of Vietnam (later the Communist Party of Vietnam) in 1951, serving as the party's chairman until his death.
    Hồ was born in Nghệ An province in French Indochina, and received a French education. Starting in 1911, he worked in various countries overseas, and in 1920 was a founding member of the French Communist Party in Paris. After studying in Moscow, Hồ founded the Vietnamese Revolutionary Youth League in 1925, which he transformed into the Indochinese Communist Party in 1930. On his return to Vietnam in 1941, he founded and led the Việt Minh independence movement against the Japanese, and in 1945 led the August Revolution against the monarchy and proclaimed the Democratic Republic of Vietnam. After the French returned to power, Hồ's government retreated to the countryside and initiated guerrilla warfare from 1946.
    Between 1953 and 1956, Hồ's leadership saw the implementation of a land reform campaign, which included executions and political purges. The Việt Minh defeated the French in 1954 at the Battle of Điện Biên Phủ, ending the First Indochina War. At the 1954 Geneva Conference, Vietnam was divided into two de facto separate states, with the Việt Minh in control of North Vietnam, and anti-communists backed by the United States in control of South Vietnam. Hồ remained president and party leader during the Vietnam War, which began in 1955. He supported the Viet Cong insurgency in the south, overseeing the transport of troops and supplies on the Ho Chi Minh trail until his death in 1969. North Vietnam won in 1975, and the country was re-unified in 1976 as the Socialist Republic of Vietnam. Saigon – Gia Định, South Vietnam's former capital, was renamed Ho Chi Minh City in his honor.
    The details of Hồ's life before he came to power in Vietnam are uncertain. He is known to have used between 50 and 200 pseudonyms. Information on his birth and early life is ambiguous and subject to academic debate. At least four existing official biographies vary on names, dates, places, and other hard facts while unofficial biographies vary even more widely. Aside from being a politician, Hồ was a writer, poet, and journalist. He wrote several books, articles, and poems in Chinese, Vietnamese, and French.
    """
    
    summary_template = """
    based on the information: {information}. From that information I want to:
    - create a short summary from that information in 2 sentences
    - give me two interesting facts about them, each fact is about 1 sentence.
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = AzureChatOpenAI(
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
        model_name=os.environ.get("AZURE_OPENAI_MODEL_NAME"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    )
    
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    response = chain.invoke({"information": information})
    
    print("\n--- Summary Result ---")
    print(response['text'])

if __name__ == "__main__":
    main()
