# app.py

import streamlit as st
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# UI Setup
st.set_page_config(page_title="Smart Email Generator", layout="centered")
st.title("📬 AI Email Generator")
st.markdown("Struggling to write the perfect email? Let AI write it beautifully for you — job apps, thank you notes, or cold outreach!")

# Inputs
role = st.text_input("Your Role", placeholder="e.g., Data Scientist")
company = st.text_input("Company", placeholder="e.g., Google")
purpose = st.selectbox("Purpose", ["Job Application", "Follow-up", "Cold Outreach", "Thank You", "Freelance Proposal"])
tone = st.selectbox("Tone", ["Professional", "Polite", "Friendly", "Persuasive"])
context = st.text_area("Extra Details", placeholder="Mention project, experience, etc.")

# Generate
if st.button("✉️ Generate Email"):
    with st.spinner("Generating email..."):
        prompt = f"""
            You are an expert email writer assistant.

            Write a well-structured, professional email for the following situation:
            - Role: {role}
            - Target Company: {company}
            - Purpose: {purpose}
            - Tone: {tone}
            - Extra Details: {context}

            The email should:
            - Start with a friendly, confident opening
            - Include at least 1 specific skill or achievement from the context
            - Mention why the applicant is interested in {company}
            - Sound like a real human wrote it
            - Include a professional sign-off

            Format the output as an email body, ready to be sent.
        """
        # Generate email using OpenAI API
        try:
            response = client.chat.completions.create(
                model="ft:gpt-4.1-2025-04-14:personal::BSQP6Qtn",  # Replace with fine-tuned model later
                messages=[{"role": "user", "content": prompt}]
            )
            email = response.choices[0].message.content.strip()
            st.success("Done!")
            st.text_area("✉️ Your AI-generated email", value=email, height=300)
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.code(email, language="markdown")
st.markdown("""
<button onclick="navigator.clipboard.writeText(document.querySelector('code').innerText)">📋 Copy to Clipboard</button>
""", unsafe_allow_html=True)

