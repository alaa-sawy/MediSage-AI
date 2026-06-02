import pandas as pd
import json
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

class MedicalInferenceEngine:
    def __init__(self):
        self.knowledge_df = pd.read_csv('expert_knowledge_base.csv')
        
        with open('severity_dict.json', 'r', encoding='utf-8') as f:
            self.severity_dict = json.load(f)
        
        self.llm = Ollama(model="llama3.2:latest", temperature=0.0)
        
        self.vectorstore = self._load_chroma()
        print("✅ Chroma DB loaded successfully!")

    def _load_chroma(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db_path = "./chroma_medical_db"
        
        if os.path.exists(db_path):
            return Chroma(persist_directory=db_path, embedding_function=embeddings)
        else:
            print("⚠️ Chroma DB not found! Please run pre.py first.")
            return None

    def _normalize_symptom(self, symptom: str) -> str:
        if not symptom:
            return ""
        s = str(symptom).strip().lower().replace('_', ' ').replace('  ', ' ')
        synonyms = {
            "stomach pain": "abdominal pain",
            "belly pain": "abdominal pain",
            "fever": "high fever",
            "mild fever": "high fever",
        }
        return synonyms.get(s, s)

    def forward_inference(self, user_symptoms: List[str]) -> List[Dict]:
        """Forward Chaining: From Symptoms → Possible Diseases"""
        if not user_symptoms:
            return []
        
        normalized_user = [self._normalize_symptom(s) for s in user_symptoms]
        query = " ".join(user_symptoms)
        
        chroma_results = self.vectorstore.similarity_search_with_score(query, k=12)
        
        formatted = []
        for doc, score in chroma_results:
            content = doc.page_content.lower()
            disease_match = None
            
            for _, row in self.knowledge_df.iterrows():
                d_name = str(row['Disease']).lower().strip()
                if d_name in content or d_name.replace(" ", "") in content.replace(" ", ""):
                    disease_match = row['Disease']
                    break
            
            if not disease_match:
                continue
                
            row = self.knowledge_df[self.knowledge_df['Disease'] == disease_match].iloc[0]
            db_syms = [s.strip().lower() for s in str(row.get('symptoms_string', '')).split(',') if s.strip()]
            
            matched = [s for s in db_syms if any(self._normalize_symptom(u) in s or s in self._normalize_symptom(u) for u in normalized_user)]
            
            match_ratio = len(matched) / len(db_syms) if db_syms else 0
            confidence = round(40 + (match_ratio * 45) + len(matched) * 12, 1)
            confidence = min(confidence, 97)
            
            if len(matched) >= 1 or confidence > 55:
                precautions = [str(row.get(f'Precaution_{i}', '')).strip() 
                              for i in range(1,5) 
                              if str(row.get(f'Precaution_{i}', '')).strip() and str(row.get(f'Precaution_{i}', '')).lower() != 'nan']
                
                formatted.append({
                    "disease": row['Disease'],
                    "confidence": confidence,
                    "match_percentage": round(match_ratio * 100, 1),
                    "matched_symptoms": matched[:5],
                    "description": str(row.get('Description', '')),
                    "precautions": precautions
                })
        
        seen = set()
        unique = [item for item in formatted if not (item['disease'] in seen or seen.add(item['disease']))]
        return sorted(unique, key=lambda x: x['confidence'], reverse=True)[:8]

    def backward_inference(self, target_disease: str, user_symptoms: List[str]) -> List[Dict]:
        """Backward Chaining: From Suspected Disease → Verify with Symptoms"""
        if not target_disease:
            return []
        
        normalized_user = [self._normalize_symptom(s) for s in user_symptoms]
        target = target_disease.strip().lower()
        
        results = []
        
        for _, row in self.knowledge_df.iterrows():
            disease_name = str(row['Disease']).lower().strip()
            
            # Check if this is the target disease (or very similar)
            if target in disease_name or disease_name in target:
                db_syms = [s.strip().lower() for s in str(row.get('symptoms_string', '')).split(',') if s.strip()]
                
                matched = [s for s in db_syms if any(self._normalize_symptom(u) in s or s in self._normalize_symptom(u) for u in normalized_user)]
                missing = [s for s in db_syms if s not in [self._normalize_symptom(m) for m in matched]]
                
                match_ratio = len(matched) / len(db_syms) if db_syms else 0
                confidence = round(match_ratio * 100, 1)
                
                precautions = [str(row.get(f'Precaution_{i}', '')).strip() 
                              for i in range(1,5) 
                              if str(row.get(f'Precaution_{i}', '')).strip() and str(row.get(f'Precaution_{i}', '')).lower() != 'nan']
                
                results.append({
                    "disease": row['Disease'],
                    "confidence": confidence,
                    "match_percentage": round(match_ratio * 100, 1),
                    "matched_symptoms": matched[:6],
                    "missing_symptoms": missing[:6],
                    "description": str(row.get('Description', '')),
                    "precautions": precautions
                })
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def generate_explanation(self, data: any, query: str, mode: str) -> str:
        if not data:
            return "### ⚠️ No Strong Match Found\nPlease provide more symptoms or try different mode."

        if mode == "ForwardChaining":
            template = """
            You are a friendly and professional doctor. Write a clear, well-organized medical report.

            Symptoms: {query}

            Use this exact format:

            # 🩺 Medical Analysis Report (Forward Chaining)

            **Symptoms Entered:** {query}

            ### 🔍 Most Likely Conditions

            {diseases}

            **Note:** This is AI-assisted analysis for educational purposes. Consult a real doctor.
            """
        else:
            template = """
            You are a friendly and professional doctor. Write a clear medical report using Backward Chaining.

            Symptoms: {query}

            Use this exact format:

            # 🩺 Medical Analysis Report (Backward Chaining)

            **Target Disease / Symptoms:** {query}

            ### 🔍 Verification Results

            {diseases}

            **Note:** This is AI-assisted analysis for educational purposes. Consult a real doctor.
            """

        diseases_text = ""
        for item in data[:6]:
            disease_name = item.get('disease', 'Unknown')
            confidence = item.get('confidence', 0)
            matched = item.get('matched_symptoms', [])
            missing = item.get('missing_symptoms', [])
            description = item.get('description', '')
            precautions = item.get('precautions', [])

            diseases_text += f"**{disease_name}** (Confidence: {confidence}%)\n\n"
            
            if matched:
                diseases_text += f"**Matched Symptoms:** {', '.join(matched)}\n"
            if missing:
                diseases_text += f"**Missing Symptoms:** {', '.join(missing)}\n"
            
            if description:
                diseases_text += f"**Description:** {description[:250]}...\n"
            
            if precautions:
                diseases_text += f"**Recommended Precautions:**\n"
                for p in precautions[:4]:
                    if p:
                        diseases_text += f"• {p}\n"
            diseases_text += "\n" + "-"*60 + "\n\n"

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "query": query,
            "diseases": diseases_text
        })