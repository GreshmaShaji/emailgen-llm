# app.py

import streamlit as st
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# UI Setup
st.set_page_config(page_title="Smart Email Generator", layout="centered")
st.title("📬 AI Email Generator")
st.markdown("Generate smart, personalized emails using GPT-3.5")

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
Category: {purpose.lower().replace(" ", "_")}
Context: A {role} writing to {company}. Tone: {tone}.
Details: {context}

Write the email:
"""
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
