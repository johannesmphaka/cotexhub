import openai
import streamlit as st
from streamlit_chat import message
import pandas as pd

# Setting page title and header
st.set_page_config(page_title="Cotexhub", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Chat to AI doctor 😬</h1>",
            unsafe_allow_html=True)

# Set org ID and API key
openai.api_key = "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"

openai_api_key_textbox = ""
model = None
tokenizer = None
generator = None
csv_name = "disease_database_mini.csv"
df = pd.read_csv(csv_name)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.subheader("Clear chats")
# model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))

st.write("Given the question, we extract keywords from the text. we focus on extracting the keywords that we can use to best lookup answers to the question\
         based on the table we used to fine-tune our models")
st.write("Question example: if I have frontal headache, fever, and painful sinuses, what disease do I have, and what  medical test should I take?")


counter_placeholder = st.sidebar.empty()
# counter_placeholder.write(
# f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")
model_name = "GPT-3.5"
# Map model names to OpenAI model IDs
# if model_name == "GPT-3.5":
#     model = "gpt-3.5-turbo"
# else:
#     model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    # counter_placeholder.write(
    #     f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append(
        {"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


def generate_response(question):

    fulltext = "A question is provided below. Given the question, extract " + \
               "keywords from the text. Focus on extracting the keywords that we can use " + \
               "to best lookup answers to the question. \n" + \
               "---------------------\n" + \
               "{}\n".format(question) + \
               "---------------------\n" + \
               "Provide keywords in the following comma-separated format.\nKeywords: "

    messages = [
        {"role": "system", "content": ""},
    ]
    messages.append(
        {"role": "user", "content": f"{fulltext}"}
    )
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    keyword_list = rsp.get("choices")[0]["message"]["content"]
    keyword_list = keyword_list.replace(",", "").split(" ")

    print(keyword_list)
    divided_text = []
    csvdata = df.to_dict('records')
    step_length = 15
    for csv_item in range(0, len(csvdata), step_length):
        csv_text = str(csvdata[csv_item:csv_item+step_length]).replace(
            "}, {", "\n\n").replace("\"", "")  # .replace("[", "").replace("]", "")
        divided_text.append(csv_text)

    answer_llm = ""

    score_textlist = [0] * len(divided_text)

    for i, chunk in enumerate(divided_text):
        for t, keyw in enumerate(keyword_list):
            if keyw.lower() in chunk.lower():
                score_textlist[i] = score_textlist[i] + 1

    answer_list = []
    divided_text = [item for _, item in sorted(
        zip(score_textlist, divided_text), reverse=True)]

    for i, chunk in enumerate(divided_text):

        if i > 4:
            continue

        fulltext = "{}".format(chunk) + \
                   "\n---------------------\n" + \
                   "Based on the Table above and not prior knowledge, " + \
                   "Select the Table Entries that will help to answer the question: {}\n Output in the format of \" Disease: <>; Symptom: <>; Medical Test: <>; Medications: <>;\". If there is no useful form entries, output: 'No Entry'".format(
                       question)

        print(fulltext)
        messages = [
            {"role": "system", "content": ""},
        ]
        messages.append(
            {"role": "user", "content": f"{fulltext}"}
        )
        rsp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer_llm = rsp.get("choices")[0]["message"]["content"]

        print("\nAnswer: " + answer_llm)
        print()
        if not "No Entry" in answer_llm:
            answer_list.append(answer_llm)

    fulltext = "The original question is as follows: {}\n".format(question) + \
               "Based on this Table:\n" + \
               "------------\n" + \
               "{}\n".format(str("\n\n".join(answer_list))) + \
               "------------\n" + \
               "Answer: "
    print(fulltext)
    messages = [
        {"role": "system", "content": ""},
    ]
    messages.append(
        {"role": "user", "content": f"{fulltext}"}
    )
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer_llm = rsp.get("choices")[0]["message"]["content"]

    print("\nFinal Answer: " + answer_llm)
    print()

    st.session_state['messages'].append(
        {"role": "assistant", "content": fulltext})

    # print(st.session_state['messages'])
    total_tokens = rsp.usage.total_tokens
    prompt_tokens = rsp.usage.prompt_tokens
    completion_tokens = rsp.usage.completion_tokens

    return answer_llm, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(
            user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            # st.write(
            #     f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            # counter_placeholder.write(
            #     f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
