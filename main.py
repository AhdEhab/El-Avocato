import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate, LLMChain
import os
from dotenv import load_dotenv
import faiss

# -------- Load API Token --------
load_dotenv()
token = os.getenv("GOOGLE_API_KEY")

# -------- Embeddings --------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        encode_kwargs={"device": "cpu"}
    )

embeddings = load_embeddings()

# -------- Load FAISS Indexes --------
constitutional_index = FAISS.load_local(
    "indexes/faiss_constitutional_index", embeddings, allow_dangerous_deserialization=True
)
statutory_index = FAISS.load_local(
    "indexes/faiss_statutory_index", embeddings, allow_dangerous_deserialization=True
)

# -------- Retrieval --------
def retrieve_docs(index, query, k=5, score_threshold=0.4):
    docs_with_scores = index.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in docs_with_scores:
        if score <= score_threshold:  
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
    if not results:
        return [{"text": "No relevant result.", "metadata": {}}]
    return results


# -------- reversed numbers --------
import re
def fix_years(text: str) -> str:
    def replace_year(match):
        year = match.group()
        # مثال: 3002 -> 2003
        if year.startswith("3") or year.startswith("0"):
            return year[::-1]  # قلب الرقم
        return year
    return re.sub(r"\b\d{4}\b", replace_year, text)

# -------- Prompt Template --------
template = """
انت نظام قانوني مصري. يجب أن تبني إجابتك على النصوص القانونية المتاحة فقط.

التعليمات:
- لا تعيد النصوص الأصلية كما هي.
- أجب بصياغة مباشرة قد السؤال فقط.
- اذكر رقم المادة + مصدرها داخل الجملة (مثال: "وفقًا للمادة 244 من قانون العقوبات").
- إذا وجدت أكثر من مادة مرتبطة، ادمجها بشكل منظم.
- إذا لم يوجد نص مناسب، اكتب (لا توجد نتيجة مرتبطة).

النصوص القانونية المتاحة (للاستخدام الداخلي فقط):
{context}

السؤال: {query}

الإجابة:
"""

llm = ChatGoogleGenerativeAI(
    model="gemma-3-4b-it",
    google_api_key=token
)

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "query"]
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
)

# -------- Query Function --------
def query_legal_assistant(query):
    constitutional_hits = retrieve_docs(constitutional_index, query, k=5, score_threshold=0.7)
    statutory_hits = retrieve_docs(statutory_index, query, k=5, score_threshold=0.7)

    def format_hits(hits, label, ref_key="article_number"):
        formatted = []
        for h in hits:
            meta = h["metadata"]
            ref = meta.get(ref_key, meta.get("page", "غير متوفر"))
            formatted.append(
                f"{label} - المادة {ref}:\n{h['text']}\n(المصدر: {meta.get('source')})"
            )
        return "\n\n".join(formatted) if formatted else f"{label}: (لا توجد نتيجة مرتبطة)"

    context = "\n\n".join([
        format_hits(constitutional_hits, "الدستور"),
        format_hits(statutory_hits, "القانون"),
    ])

    response = llm_chain.run({"context": context, "query": query})
    response = fix_years(response)
    return response

# -------- Streamlit UI --------
st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️", layout="wide")

st.image("D:\Digitopia\App\logo.jpg", width=250) 
st.write("اسأل سؤالاً قانونياً وسيتم استخراج النصوص الدستورية والتشريعية ذات الصلة.")

query = st.text_area("📝 أدخل سؤالك:", placeholder="مثال: ما هي الإجراءات المطلوبة لتقديم طلب إخلاء سبيل متهم؟")

if st.button("إرسال الاستعلام"):
    if query.strip():
        with st.spinner("⏳ جاري البحث وتحضير الإجابة..."):
            answer = query_legal_assistant(query)

        st.subheader(" الإجابة:")
        st.write(answer)
    else:
        st.warning("⚠️ الرجاء إدخال سؤال أولاً.")
