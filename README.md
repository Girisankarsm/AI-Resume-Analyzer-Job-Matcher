🧠 AI Resume Analyzer & Job Matcher

An interactive Streamlit web app that helps job seekers analyze their resume against a job description.
It extracts skills, computes similarity, highlights missing skills, and gives tailored suggestions to improve your resume.

🚀 Features

📂 Upload resume in PDF/DOCX/TXT

📝 Paste job description text

📊 Get a cosine similarity match score

✅ Detect matching skills and ❗ highlight missing skills

🔎 Show key overlapping terms from resume & JD

🎯 Skill coverage heatmap visualization

💡 Generate resume bullet suggestions based on missing skills

⬇️ Export results as JSON summary

🛠️ Tech Stack

Streamlit
 – UI framework

scikit-learn
 – TF-IDF vectorization & similarity

Matplotlib
 & Seaborn
 – Data visualization

PyPDF2
, python-docx
 – Resume parsing

📦 Installation
# Clone this repo
git clone https://github.com/yourusername/resume-analyzer.git
cd resume-analyzer

# Create environment & install dependencies
pip install -r requirements.txt

▶️ Usage
streamlit run app.py


Then open http://localhost:8501
 in your browser.

📌 Example Output

Match Score: 78.5%

Matching Skills: Python, Pandas, SQL

Missing Skills: TensorFlow, Kubernetes, CI/CD

Heatmap: Visual skill coverage between resume & JD

Suggested Bullet: "Implemented TensorFlow models in production; improved prediction accuracy by X%."
