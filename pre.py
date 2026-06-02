import pandas as pd
import json
import shutil
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_symptom(sym: str) -> str:
    """Clean and normalize symptom text"""
    if not isinstance(sym, str):
        return ""
    s = str(sym).strip().lower().replace('_', ' ').replace('  ', ' ')
    s = s.replace('dischromic _patches', 'dischromic patches')
    s = s.replace('spotting_ urination', 'spotting urination')
    s = s.replace('foul_smell_of urine', 'foul smell of urine')
    return s


def run_preprocessing():
    """Build Knowledge Base and VectorStore for LangChain"""
    
    print("🚀 Starting knowledge base build process...")

    # Clear old Chroma database
    if os.path.exists("./chroma_medical_db"):
        shutil.rmtree("./chroma_medical_db")
        print("🗑️ Old Chroma database cleared.")

    # Check where the data files are
    if os.path.exists("DB/dataset.csv"):
        data_path = "DB/"
        print("📂 Reading data from DB/ folder")
    else:
        data_path = ""
        print("📂 Reading data from root folder")

    # Load datasets
    df = pd.read_csv(f'{data_path}dataset.csv')
    severity = pd.read_csv(f'{data_path}Symptom-severity.csv')
    description = pd.read_csv(f'{data_path}symptom_Description.csv')
    precaution = pd.read_csv(f'{data_path}symptom_precaution.csv')

    symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]
    df[symptom_cols] = df[symptom_cols].fillna('')

    # Clean symptoms
    df['symptoms_list'] = df[symptom_cols].apply(
        lambda row: [clean_symptom(s) for s in row if clean_symptom(s)], axis=1
    )

    # Build aggregated knowledge base
    df_kb = df.groupby('Disease').agg(
        symptoms_list=('symptoms_list', lambda x: list(set(sum(x, []))))
    ).reset_index()

    df_kb = df_kb.merge(description, on='Disease', how='left')
    df_kb = df_kb.merge(precaution, on='Disease', how='left')
    df_kb['symptoms_string'] = df_kb['symptoms_list'].apply(lambda x: ', '.join(x))

    # Save output files
    df_kb.to_csv('expert_knowledge_base.csv', index=False)
    severity.to_csv('cleaned_severity.csv', index=False)

    # Save severity dictionary
    severity_dict = {clean_symptom(row['Symptom']): int(row['weight']) 
                     for _, row in severity.iterrows()}
    with open('severity_dict.json', 'w', encoding='utf-8') as f:
        json.dump(severity_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ Knowledge Base created successfully! ({len(df_kb)} diseases)")

    # ====================== Create Chroma VectorStore ======================
    print("🔄 Creating Chroma Vector Database... (this may take a minute)")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    documents = []
    for _, row in df_kb.iterrows():
        text = f"""
        Disease: {row['Disease']}
        Symptoms: {row['symptoms_string']}
        Description: {row.get('Description', '')}
        """
        documents.append(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.create_documents(documents)

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_medical_db"
    )

    print("🎉 Chroma Vector Database created successfully!")
    print(f"📁 Path: {os.path.abspath('./chroma_medical_db')}")


if __name__ == "__main__":
    run_preprocessing()