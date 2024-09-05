from groq import Groq
import streamlit as st

st.title("ChatGPT-like clone with Groq API")

# Initialize Groq client
client = Groq(api_key=st.secrets["groq"]["api_key"])

# Ensure session state holds the necessary information
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama-3.1-8b-instant"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input and process it
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from Groq
    with st.chat_message("assistant"):
        completion = client.chat.completions.create(
            model=st.session_state["groq_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content.strip()
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Append assistant's response to the session state
    # st.session_state.messages.append({"role": "assistant", "content": response})
    
