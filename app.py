import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --------------------------
# Load the model once
# --------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --------------------------
# ✅ NEW NAME!
# --------------------------
st.title("🤖 MUVERA PassageIQ AI: SEO Content Analyzer")

st.markdown("""
Welcome to **MUVERA PassageIQ AI**, your smart assistant for passage-level SEO analysis.  
This tool breaks your content into **passages** (~50–150 words each).  
If you enter a **search intent/query**, it checks how **retrievable** each passage is using semantic similarity.

**🔑 What is Retrievability Score?**  
It shows how closely a passage matches your query.  
- **1.0** → Highly relevant  
- **0.0** → Not relevant

---

**💡 Tip:** Paste your content, add a query if you want, then click **Analyze Content**.
""")

# --------------------------
# User Inputs
# --------------------------
content = st.text_area(
    "📄 Paste your content here",
    height=300,
    placeholder="Paste your SEO draft or article here..."
)

query = st.text_input(
    "🔎 Optional: Enter a search intent/query (e.g., 'best credit cards for students')"
)

# --------------------------
# Analyze Button
# --------------------------
if st.button("🚀 Analyze Content"):
    if not content.strip():
        st.warning("⚠️ Please paste some content to analyze.")
    else:
        with st.spinner("🔄 Analyzing content..."):
            # --------------------------
            # Split content into passages (~50 words for better granularity)
            # --------------------------
            words = content.split()
            chunk_size = 50  # Use 50 for shorter content, 150 for long pages
            passages = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

            # --------------------------
            # Get embeddings & similarity scores
            # --------------------------
            embeddings = model.encode(passages)
            similarity_scores = [None] * len(passages)

            if query.strip():
                query_embedding = model.encode([query])
                similarity_scores = cosine_similarity(query_embedding, embeddings)[0]

            # --------------------------
            # Build DataFrame
            # --------------------------
            df = pd.DataFrame({
                "Passage #": [f"Passage {i+1}" for i in range(len(passages))],
                "Text": passages,
                "Retrievability Score": [
                    round(float(score), 3) if score is not None else None for score in similarity_scores
                ]
            })

        st.success("✅ Analysis complete!")

        # --------------------------
        # Display Full Table
        # --------------------------
        st.subheader("📊 Full Analysis Results")
        st.dataframe(df, use_container_width=True)

        if query.strip():
            # --------------------------
            # Top Passages
            # --------------------------
            st.subheader("🏆 Top Passages (High Relevance)")
            st.markdown("""
These are the passages most relevant to your query, sorted by **Retrievability Score** (1 = highly relevant).
""")
            top_df = df.sort_values(by="Retrievability Score", ascending=False).head(3)

            for _, row in top_df.iterrows():
                st.markdown(f"**✅ {row['Passage #']}** | **Score:** {row['Retrievability Score']}")
                st.info(row['Text'])
                st.caption("🔍 This passage strongly matches your query. Use it for snippets, intros, or meta descriptions.")
                st.write("---")

            # --------------------------
            # Weak Passages
            # --------------------------
            if len(df) >= 4:
                st.subheader("⚠️ Weak Passages (Low Relevance)")
                st.markdown("""
These passages are least relevant to your query.  
Consider rewriting them to better match the intent.
""")
                weak_df = df.sort_values(by="Retrievability Score", ascending=True).head(3)
                # Remove overlaps with Top Passages
                weak_df = weak_df[~weak_df["Passage #"].isin(top_df["Passage #"])]
                if not weak_df.empty:
                    for _, row in weak_df.iterrows():
                        st.markdown(f"**⚠️ {row['Passage #']}** | **Score:** {row['Retrievability Score']}")
                        st.warning(row['Text'])
                        st.caption("✏️ This passage may not match the intent well. Add related keywords or rewrite it for better alignment.")
                        st.write("---")
                else:
                    st.info("✅ All passages are relevant — no weak passages to rewrite!")
            else:
                st.info("ℹ️ Not enough passages to show weak ones. Add more content for better analysis.")

            # --------------------------
            # Score Chart
            # --------------------------
            st.subheader("📈 Retrievability Score Chart")
            st.bar_chart(df.set_index("Passage #")["Retrievability Score"])

        else:
            st.info("ℹ️ You didn't enter a query — retrievability scores not calculated. Add a query and re-run!")

        # --------------------------
        # Download CSV
        # --------------------------
        st.download_button(
            "📥 Download Results as CSV",
            df.to_csv(index=False),
            "muvera_results.csv",
            "text/csv"
        )

        st.info("💡 *Future versions could auto-rewrite weak passages using GPT.*")
