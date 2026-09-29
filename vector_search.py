import os
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

client = genai.Client(api_key=api_key)

# Create a persistent local Chroma database
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection
collection = chroma_client.get_or_create_collection(
    name="university_faqs"
)

# Sample university FAQs
documents = [
    "Students can apply for admission online through the university admissions portal.",
    "The computer science program covers programming, databases, software engineering, and artificial intelligence.",
    "Students can contact the academic office for help with course registration and academic matters.",
    "The university library provides access to textbooks, research papers, and online academic resources.",
    "Students can apply for scholarships through the financial aid office before the application deadline.",
    "The university provides student accommodation for eligible students near the main campus.",
    "Examinations are normally held at the end of each academic semester.",
    "Students can join university clubs and societies to participate in extracurricular activities.",
    "The university offers career support including internships, job preparation, and career counseling.",
    "Students can contact the IT support team for help with university accounts and technical problems.",
]

metadata = [
    {"category": "admissions", "title": "Online Admission", "source": "University FAQ"},
    {"category": "academics", "title": "Computer Science Program", "source": "University FAQ"},
    {"category": "academics", "title": "Academic Office", "source": "University FAQ"},
    {"category": "library", "title": "University Library", "source": "University FAQ"},
    {"category": "financial_aid", "title": "Scholarships", "source": "University FAQ"},
    {"category": "student_life", "title": "Student Accommodation", "source": "University FAQ"},
    {"category": "examinations", "title": "Semester Examinations", "source": "University FAQ"},
    {"category": "student_life", "title": "Clubs and Societies", "source": "University FAQ"},
    {"category": "careers", "title": "Career Support", "source": "University FAQ"},
    {"category": "technical_support", "title": "IT Support", "source": "University FAQ"},
]


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


print("Generating embeddings and storing information in Chroma...")

for index, (document, document_metadata) in enumerate(zip(documents, metadata)):
    embedding = get_embedding(document)

    collection.upsert(
        ids=[f"faq-{index + 1}"],
        documents=[document],
        embeddings=[embedding],
        metadatas=[document_metadata],
    )

print(f"Stored {len(documents)} records in Chroma.")
print("Vector database setup completed successfully.")
print("\nStarting semantic search...\n")

questions = [
    "How can I get financial help for my studies?",
    "Where can I get help with my computer account?",
    "What support is available for finding internships and jobs?",
]


for question in questions:
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
    )

    print("=" * 70)
    print("Question:", question)
    print("\nMost relevant results:")

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"- Distance: {distance:.4f}")
        print(f"  Category: {metadata['category']}")
        print(f"  Title: {metadata['title']}")
        print(f"  Information: {document}")
        print()