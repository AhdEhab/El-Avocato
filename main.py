import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate, LLMChain
import os
from dotenv import load_dotenv

# -------- Load API Token --------
load_dotenv()
token = os.getenv("GOOGLE_API_KEY")

# -------- Embeddings --------
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# -------- Load FAISS Indexes --------
constitutional_index = FAISS.load_local(
    "indexes/faiss_constitutional_index", embeddings, allow_dangerous_deserialization=True
)
statutory_index = FAISS.load_local(
    "indexes/faiss_statutory_index", embeddings, allow_dangerous_deserialization=True
)

# -------- Retrieval (بدون threshold) --------
def retrieve_docs(index, query, k=5):
    docs = index.similarity_search("query: " + query, k=k)
    results = []
    for doc in docs:
        results.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    if not results:
        return [{"text": "No relevant result.", "metadata": {}}]
    return results

# -------- Prompt Template --------
template = """
انت نظام قانوني مصري. يجب أن تبني إجابتك على طبقتين مترابطتين هرمياً:
1. الدستور (القواعد العليا) → يضع المبادئ العامة.
2. التشريعات (القوانين المفسرة) → تستند إلى الدستور وتفسر مواده.

التعليمات:
- اعرض النصوص الدستورية أولاً إن وُجدت.
- ثم اعرض النصوص التشريعية المرتبطة بها.
- لكل جزء، اذكر النص كاملاً مع المصدر ورقم المادة من الـ metadata.
- إذا لم يوجد نص في طبقة معينة، اكتب (لا توجد نتيجة مرتبطة).
- لا تضف أي نص أو تفسير غير موجود في قاعدة البيانات.

السياق القانوني (مرتّب ومترابط وليس مقاطع منفصلة):
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
    constitutional_hits = retrieve_docs(constitutional_index, query, k=5)
    statutory_hits = retrieve_docs(statutory_index, query, k=5)

    def format_hits(hits, label, ref_key="article_number"):
        if hits and hits[0]["text"] != "No relevant result.":
            formatted = []
            for h in hits:
                meta = h["metadata"]
                ref = meta.get(ref_key, meta.get("page", "غير متوفر"))
                formatted.append(
                    f"النص: {h['text']}\n"
                    f"المصدر: {meta.get('source')}, المرجع: {ref}"
                )
            return f"## {label}\n" + "\n\n".join(formatted)
        else:
            return f"## {label}\n(لا توجد نتيجة مرتبطة)"

    context = "\n\n".join([
        format_hits(constitutional_hits, "الدستور (القواعد العليا)", "article_number"),
        format_hits(statutory_hits, "التشريعات (القوانين المفسرة)", "article_number"),
    ])

    response = llm_chain.run({"context": context, "query": query})
    return response, context


# -------- Streamlit UI --------
st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️", layout="wide")

st.title("⚖️ المساعد القانوني المصري")
st.write("اسأل سؤالاً قانونياً وسيتم استخراج النصوص الدستورية والتشريعية ذات الصلة.")

query = st.text_area("📝 أدخل سؤالك:", placeholder="مثال: ما هي حقوق حرية التعبير في الدستور المصري؟")

if st.button("إرسال الاستعلام"):
    if query.strip():
        with st.spinner("⏳ جاري البحث وتحضير الإجابة..."):
            answer, context = query_legal_assistant(query)

        st.subheader("📌 الإجابة:")
        st.write(answer)

        with st.expander("📂 السياق المستخدم (النصوص المسترجعة)"):
            st.markdown(context)
    else:
        st.warning("⚠️ الرجاء إدخال سؤال أولاً.")

