document.getElementById("generate").addEventListener("click", async () => {
    const role = document.getElementById("role").value;
    const company = document.getElementById("company").value;
    const purpose = document.getElementById("purpose").value;
    const context = document.getElementById("context").value;
  
    const prompt = `
      You are an expert email writer assistant.
  
      Write a well-structured, professional email for:
      - Role: ${role}
      - Company: ${company}
      - Purpose: ${purpose}
      - Context: ${context}
  
      Include strong opening, relevant experience, company alignment, and a human tone.
    Make it concise and engaging.
    `;
  
    try {
      const response = await fetch("https://emailgen-llm.streamlit.app/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt })
      });
  
      const data = await response.json();
      document.getElementById("result").textContent = data.output || "No response.";
    } catch (err) {
      document.getElementById("result").textContent = "Error: " + err.message;
    }
  });
  