import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Multilingual Semantic Search",
    layout="centered"
)

SAMPLE_DOCUMENT = """
गोरिल्ला बड़े और मुख्य रूप से शाकाहारी महान कपि होते हैं जो भूमध्यरेखीय अफ्रीका के उष्णकटिबंधीय जंगलों में रहते हैं।
गोरिल्ला वंश को पूर्वी और पश्चिमी प्रजातियों में विभाजित किया गया है।

गोरिल्ला वर्तमान में जीवित सबसे बड़े प्राइमेट होते हैं और आमतौर पर समूहों में रहते हैं, जिनका नेतृत्व एक सिल्वरबैक करता है।
वे अपनी बुद्धिमत्ता और मजबूत सामाजिक संबंधों के लिए जाने जाते हैं।

गोरिल्ला उप-सहारा अफ्रीका के विभिन्न ऊँचाई वाले क्षेत्रों में पाए जाते हैं।
आवास विनाश और अवैध शिकार के कारण गोरिल्ला की कई प्रजातियाँ संकटग्रस्त हैं।
"""

st.title("🌐 Multilingual Semantic Search")
st.write(
    "Upload or paste documents and search them by meaning, "
    "not keywords. Works across languages."
)

doc_mode = st.radio("Document Input", ["Paste text", "Upload .txt file"])

show_sample = st.checkbox("Show sample document")

if show_sample:
    st.text_area("Sample document", SAMPLE_DOCUMENT, height=200, disabled=True)

if st.button("Load sample document"):
    try:
        response = requests.post(f"{API_URL}/index/text", json={"text": SAMPLE_DOCUMENT})
        if response.status_code == 200:
            st.session_state.indexed = True
            st.success(response.json()["message"])
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")

query = st.text_input("Enter your query", placeholder="e.g. What is a gorilla?")

use_rag = st.checkbox("Answer using retrieved context (RAG)")
top_k = st.slider("Number of results", 1, 5, 3)

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if doc_mode == "Paste text":
    user_text = st.text_area("Paste your document text", height=200)
    if st.button("Index text") and user_text.strip():
        try:
            response = requests.post(f"{API_URL}/index/text", json={"text": user_text})
            if response.status_code == 200:
                st.session_state.indexed = True
                st.success(response.json()["message"])
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file and st.button("Index document"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            response = requests.post(f"{API_URL}/index/file", files=files)
            if response.status_code == 200:
                st.session_state.indexed = True
                st.success(response.json()["message"])
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

if st.button("Search") and query.strip():
    if not st.session_state.indexed:
        st.warning("Please paste, upload, or load the sample document first")
    else:
        payload = {
            "query": query,
            "top_k": top_k,
            "use_rag": use_rag
        }
        with st.spinner("Searching..."):
            try:
                response = requests.post(f"{API_URL}/search", json=payload)
                
                if response.status_code != 200:
                    st.error(f"Error: {response.text}")
                    st.stop()
                    
                data = response.json()
                results = data.get("results", [])
                answer = data.get("answer")
                
                if not results:
                    st.warning("No relevant results found")
                    st.stop()

                if use_rag and answer:
                    st.subheader("Answer")
                    st.write(answer)
                elif use_rag and not answer:
                    st.info("RAG is enabled but answer generation failed or GROQ_API_KEY is missing on the server.")

                st.subheader("Results")
                for i, res in enumerate(results, 1):
                    st.markdown(f"{i}.")
                    st.write(res["text"])
                    st.caption(f"Similarity: {res['score']:.3f}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
