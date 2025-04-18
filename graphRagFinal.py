# GraphRAG imports 
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
# from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import uuid
from typing import List, Dict, Tuple, Any
import numpy as np
import json
import ast
import openai  # or any LLM wrapper
import random  # for dummy confidence
import spacy
import os
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from llama_index.core import ServiceContext
import matplotlib.pyplot as plt
import heapq
from typing import List, Dict, Any
from difflib import SequenceMatcher
import logging
import json
from openai import OpenAI

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunker = SemanticSplitterNodeParser(embed_model=embed_model, chunk_size=5)  # adjust chunk size as needed

# Step 1: Sentence-level split
sentence_splitter = SentenceSplitter()
# semantic chunk + sentence splitting 
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def hybrid_semantic_chunk(text):
    # Step 1: Sentence split using nltk
    sentence_texts = sent_tokenize(text)

    print(f"\n📘 Sentence Split ({len(sentence_texts)} segments):")
    for s in sentence_texts:
        print(f"- {s}")

    # Step 2: Apply semantic chunking to long sentences
    final_chunks = []
    for sentence in sentence_texts:
        if len(sentence.split()) > 20:
            deeper_chunks = chunker.get_nodes_from_documents([TextNode(text=sentence)])
            final_chunks.extend([n.text for n in deeper_chunks])
        else:
            final_chunks.append(sentence)

    if len(final_chunks) > 1:
        print("\n🔹 Original Text:")
        print(text)
        print("🔹 Split Into:")
        for ch in final_chunks:
            print(f"- {ch}")

    return final_chunks

# Semantic chunking ONLY, splitter
def semantic_chunk(text):
    doc = TextNode(text=text)
    nodes = chunker.get_nodes_from_documents([doc])
    chunks = [node.text for node in nodes]

    # DEBUG
    if len(chunks) > 1:
        print("\n🔹 Original Text:")
        print(text)
        print("🔹 Split Into:")
        for ch in chunks:
            print(f"- {ch}")
    
    return chunks

# Main transcript processor with chunking applied
def preprocess_transcript(transcript: str):
    lines = transcript.strip().split("\n")
    structured = []

    speaker_line_pattern = re.compile(r"^(Telemarketer|Customer):\s*(.*)")

    for line in lines:
        match = speaker_line_pattern.match(line)
        if match:
            speaker, text = match.groups()
            chunks = hybrid_semantic_chunk(text.strip()) # hybrid_semantic_chunk(text.strip()) 

            for chunk in chunks:
                structured.append({
                    "speaker": speaker,
                    "text": chunk.strip()
                })

    return structured

def print_preprocessed_transcript(preprocessed):
    print("\n=============== Preprocessed Transcript ===============")
    for i, entry in enumerate(preprocessed, start=1):
        print(f"🔹 Chunk {i}")
        print(f"👤 Speaker: {entry['speaker']}")
        print(f"🗣️ Text: {entry['text']}")
        print("-" * 50)


doc = """
Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?

Customer: Yeah, sure.

Telemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?

Customer: Elena, no.

Telemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.

Customer: This is regarding what?

Telemarketer: Yeah, so very quickly, I'm from IPP Financial Advisors, so basically we are Singapore's largest and oldest financial advisory firm, and we help expats and professionals specifically, like yourself, with everything from offshore investments, retirement planning, university fee planning, insurance, offshore investments as well.

Telemarketer: So just wanted to check with you if you might be keen to, you know, jump on a 15 to 20 minute Zoom session, just a sharing session with yourself and Elena Pryor, who's our senior consultant, maybe next week or the week after?

Customer: Yeah, I'm not interested right now in making any investments, so that's the reason maybe I have not responded.

Telemarketer: I see. That's ok. Maybe if you just want to learn more and we can exchange business cards as well, that way we can stay in touch whenever you might be more keen and more interested to.

Customer: Sure, we can set it up for some time on Saturday.

Telemarketer: Yeah, Saturday is fine. So would you be interested on, let's say, 9th of March, which is a Saturday?

Customer: That should be ok. We can set up some time in the late afternoon.

Telemarketer: Ok, so late afternoon, would 12 p.m. be ok with you?

Customer: Should be ok, yeah.

Telemarketer: Should be ok, alright. What I can do is tentatively set a meeting for 12 p.m. on 9th March and closer to that date, maybe I can drop you a follow-up.

Customer: Sure, sure, sure.

Telemarketer: Ok, thanks. And your email, maybe I can send you a Zoom invite via the email as well.

Customer: Yeah, you can send me over the text, right? This is a WhatsApp number, you can send me over there on WhatsApp.

Telemarketer: Alright, no worries. Thank you.

Customer: Thanks.

Telemarketer: Thank you
"""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai.api_key = OPENAI_API_KEY
llm = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)
# Optional spaCy fallback
# Extracts root verbs and main objects using dependency tags.
nlp = spacy.load("en_core_web_sm")

# FACT SUMMARY METADATA 
def generate_fact_summary(text: str, llm=False) -> str:
    if llm:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "Extract all the key actions or intentions expressed in the sentence. "
                        "Respond with a concise sentence fragment or list (e.g., 'Introduced self, proposed a meeting', "
                        "'Acknowledged understanding', 'Asked a clarifying question'). "
                        "Avoid general topics — focus only on specific speaker actions or intents."
                    )},
                    {"role": "user", "content": text}
                ],
                max_tokens=50,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ LLM failed for fact summary: {text} — {str(e)}")
            return rule_based_fact_summary(text)
    else:
        return rule_based_fact_summary(text)


def detect_topics(text: str, llm=False) -> List[str]:
    if llm:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Identify high-level topics or themes in the following sentence. Return them as a comma-separated list."},
                    {"role": "user", "content": text}
                ],
                max_tokens=50,
                temperature=0
            )
            topics_str = clean_text(response.choices[0].message.content.strip())
            return [t.strip().lower() for t in topics_str.split(",") if t.strip()]
        except Exception as e:
            print(f"⚠️ LLM failed for topics: {text} — {str(e)}")
            return rule_based_detect_topics(text)
    else:
        return rule_based_detect_topics(text)


def detect_details(text: str, llm=False) -> List[str]:
    if llm:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "List any contextual or delivery-related details in this sentence, such as tone, language clarity, or interaction style. Use short phrases, comma-separated."},
                    {"role": "user", "content": text}
                ],
                max_tokens=60,
                temperature=0
            )
            details_str = clean_text(response.choices[0].message.content.strip())
            return [d.strip() for d in details_str.split(",") if d.strip()]
        except Exception as e:
            print(f"⚠️ LLM failed for details: {text} — {str(e)}")
            return rule_based_detect_details(text)
    else:
        return rule_based_detect_details(text)
    
# RULE-BASED EXTRACTOR FUNCTIONS (Keyword Mappings from Audit Criteria)
def rule_based_fact_summary(text: str) -> str:
    doc = nlp(text)
    actions = set()
    text_lower = text.lower()

    for sent in doc.sents:
        lemmas = [token.lemma_ for token in sent]
        deps = [token.dep_ for token in sent]
        sent_text = sent.text.lower()

        # Introduced self
        if any(t.lemma_ in ["call", "introduce", "name"] and t.dep_ in ["ROOT", "attr"] for t in sent) and ("from" in sent_text or "this is" in sent_text):
            actions.add("Introduced self")

        # Stated company
        if any(org in sent_text for org in ["ipp", "ippfa", "ipp financial advisors"]) and "on behalf" not in sent_text:
            actions.add("Stated company")

        # Asked about contact source
        if re.search(r"(how.*get.*number|who.*gave.*(number|contact))", sent_text) or any("contact" in token.text.lower() for token in sent):
            actions.add("Asked about contact source")

        # Explained services
        if any(l in lemmas for l in ["invest", "plan", "save", "return", "guarantee", "insurance", "cover"]):
            actions.add("Explained services")

        # Proposed meeting
        if any(l in lemmas for l in ["schedule", "arrange", "book", "meet", "zoom", "set"]):
            if "meeting" in sent_text or "zoom" in sent_text:
                actions.add("Proposed meeting")
                
        # Proposed meeting (expanded)
        if re.search(r"(join.*(zoom|call|meeting)|hop on a call|talk to consultant|schedule a session)", sent_text):
            actions.add("Proposed meeting")

        # Mentioned guaranteed returns
        if "high return" in sent_text or "guarantee" in sent_text or "capital guarantee" in sent_text:
            actions.add("Mentioned guaranteed returns")

        # Professional tone
        if any(p in sent_text for p in ["thank you", "no worries", "sure", "alright", "okay", "i see", "that’s ok", "understood"]):
            actions.add("Professional tone")

        # Checked interest
        if "keen to explore" in sent_text or re.search(r"are you (still )?(interested|keen|open)", sent_text):
            actions.add("Checked interest")

        # Customer uncertainty
        if re.search(r"not interested|maybe later|not now|not sure|unsure|don’t think so", sent_text):
            actions.add("Customer uncertainty")

        # Pressured appointment
        if "not interested" in sent_text and re.search(r"(set.*meeting|zoom|schedule)", sent_text):
            actions.add("Pressured appointment")
        
        # Add detection for "Who gave your number" responses
        if re.search(r"(my (manager|consultant|colleague).*(gave|shared).*(your|the).*contact)", sent_text):
            actions.add("Explained contact source")
        
        # Explained contact source
        if re.search(r"(my (manager|consultant|colleague).*(gave|shared).*(your|the).*contact)", sent_text):
            actions.add("Explained contact source")

    return ", ".join(sorted(actions)) or "No key actions found."


def rule_based_detect_topics(text: str) -> List[str]:
    doc = nlp(text)
    topics = set()
    text_lower = text.lower()

    for sent in doc.sents:
        lemmas = [token.lemma_ for token in sent]

        if any(l in lemmas for l in ["introduce", "call", "name"]) and "from" in sent.text.lower():
            topics.add("introduction")
        if any(org in sent.text.lower() for org in ["ipp", "ippfa", "financial advisor"]):
            topics.add("company disclosure")
        if "how did you get" in sent.text.lower() or "who gave" in sent.text.lower():
            topics.add("lead source")
        if any(l in lemmas for l in ["invest", "return", "insurance", "plan", "cover"]):
            topics.add("financial services")
        if "meeting" in sent.text.lower() or "zoom" in sent.text.lower():
            topics.add("meeting setup")
        if "high return" in sent.text.lower() or "guarantee" in sent.text.lower():
            topics.add("product claim")
        if "not interested" in sent.text.lower() or "maybe later" in sent.text.lower() or "unsure" in sent.text.lower():
            topics.add("customer uncertainty")
        if "keen to explore" in sent.text.lower() or "would you be interested" in sent.text.lower():
            topics.add("interest probe")
        if "not interested" in sent.text.lower() and "meeting" in sent.text.lower():
            topics.add("pressure tactics")

    return sorted(topics) or ["general"]


def rule_based_detect_details(text: str) -> List[str]:
    doc = nlp(text)
    details = set()
    text_lower = text.lower()

    if len(text.split()) < 8:
        details.add("Concise")

    if "?" in text:
        details.add("Asked a question")

    polite_phrases = ["thank you", "no worries", "sure", "ok", "okay", "alright"]
    if any(p in text_lower for p in polite_phrases):
        details.add("Friendly tone")
        
    polite_expressions = ["thank you", "thanks", "appreciate it", "sure", "ok", "no worries", "alright", "understood"]
    if any(phrase in text_lower for phrase in polite_expressions):
        details.add("Friendly tone")

    if re.search(r"not interested|maybe later|not now|unsure|don’t think so", text_lower):
        details.add("Customer hesitance")

    # Tone detection via POS tags or dependency (very basic)
    if any(token.lemma_ == "apologize" or token.text.lower() == "sorry" for token in doc):
        details.add("Apologetic")

    return sorted(details) or ["No details found"]

# Full enrichment step ( FACT SUMMARY + DETAIL + TOPIC METDATA )
def enrich_chunks(chunks: List[Dict], llmTF = False) -> List[Dict]:
    enriched = []

    for i, chunk in enumerate(chunks, start=1):
        node_id = f"n{i}"
        text = chunk["text"]
        speaker = chunk["speaker"]

        print(f"\n🔄 Processing Chunk {node_id} ({speaker}):")
        print(f"💬 Text: {text}")

        print("   ➤ Generating 📌 Fact Summary...")
        fact_summary = generate_fact_summary(text, llm=llmTF)

        print("   ➤ Generating 🧩 Details...")
        details = detect_details(text, llm=llmTF)

        print("   ➤ Generating 🏷️  Topics...")
        topics = detect_topics(text, llm=llmTF)
        
        #Compute confidence using cosine similarity
        try:
            text_embedding = embed_model.get_text_embedding(text)
            summary_embedding = embed_model.get_text_embedding(fact_summary)
            similarity = cosine_similarity(
                np.array(text_embedding).reshape(1, -1),
                np.array(summary_embedding).reshape(1, -1)
            )[0][0]
            confidence_score = round(float(similarity), 3)
        except Exception as e:
            print(f"⚠️ Failed to compute confidence: {e}")
            confidence_score = 0.0

        enriched.append({
            "id": node_id,
            "speaker": speaker,
            "original_text": text,
            "fact_summary": fact_summary,
            "details": details,
            "topics": topics,
            "confidence": confidence_score
        })

    return enriched

def clean_text(text: str) -> str:
    # Fix broken words caused by newlines or wrap issues
    # e.g. "familia\n rity" → "familiarity", but keep real word spacing
    text = re.sub(r'(\w+)[\n\r]\s+(\w+)', r'\1\2', text)  # Join words split by newline + indent
    text = re.sub(r'(\w+)\s{2,}(\w+)', r'\1 \2', text)     # Normalize accidental double spaces
    return text.strip()

def print_enriched_nodes(enriched_nodes):
    for node in enriched_nodes:
        print(f"🟦 ID: {node['id']}")
        print(f"👤 Speaker: {node['speaker']}")
        print(f"🗣️ Original Text: {node['original_text']}")
        print(f"📌 Fact Summary: {node['fact_summary']}")
        print(f"🧩 Details: {', '.join(node['details'])}")
        print(f"🏷️ Topics: {', '.join(node['topics'])}")
        print(f"🎯 Confidence: {node['confidence']}")
        print("-" * 60)

# raw_chunks = preprocess_transcript(doc)
# print_preprocessed_transcript(raw_chunks)
# enriched_nodes = enrich_chunks(raw_chunks, llmTF=True)
# print("\n=============== Enriched Transcript Nodes ===============")
# print_enriched_nodes(enriched_nodes)

# ------------------------------Node & Edge Construction--------------------------------

# # Step 1: Get all chunk texts
# chunk_texts = [node["original_text"] for node in enriched_nodes]

# # Step 2: Generate embeddings
# embeddings = embed_model.get_text_embedding_batch(chunk_texts)  # returns List[List[float]]

# # Step 3: Create nodes
# nodes = [
#     {
#         "id": node["id"],
#         "speaker": node["speaker"],
#         "text": node["original_text"],
#         "fact_summary": node["fact_summary"],
#         "topics": node["topics"],
#         "details": node["details"],
#         "confidence": node["confidence"]
#     }
#     for node in enriched_nodes
# ]

# # Step 4: Build edges
# edges = []

# for i in range(len(enriched_nodes) - 1):
#     curr = enriched_nodes[i]
#     next_node = enriched_nodes[i + 1]

#     curr_id = curr["id"]
#     next_id = next_node["id"]

#     # ➤ Sequential edge (default response type)
#     edges.append({
#         "from": curr_id,
#         "to": next_id,
#         "type": "response"
#     })

#     # ➤ Continuation (same speaker)
#     if curr["speaker"] == next_node["speaker"]:
#         edges.append({
#             "from": curr_id,
#             "to": next_id,
#             "type": "continuation"
#         })

#     # ➤ Semantic similarity with topic overlap
#     sim = cosine_similarity(
#         np.array(embeddings[i]).reshape(1, -1),
#         np.array(embeddings[i + 1]).reshape(1, -1)
#     )[0][0]

#     shared_topics = set(curr["topics"]) & set(next_node["topics"])
#     if shared_topics and sim >= 0.75:
#         edges.append({
#             "from": curr_id,
#             "to": next_id,
#             "type": "related_topic",
#             "similarity": round(float(sim), 4)
#         })

# CONSTRUCT NODES
def construct_nodes(enriched_nodes, embed_model):
    print("\n=============== Constructing Nodes ===============")
    chunk_texts = [node["original_text"] for node in enriched_nodes]
    embeddings = embed_model.get_text_embedding_batch(chunk_texts)
    nodes = []
    for node in enriched_nodes:
        node_data = {
            "id": node["id"],
            "speaker": node["speaker"],
            "text": node["original_text"],
            "fact_summary": node["fact_summary"],
            "topics": node["topics"],
            "details": node["details"],
            "confidence": node["confidence"]
        }
        nodes.append(node_data)
        print(f"\n🟢 Node: {node_data['id']}")
        print(f"👤 Speaker: {node_data['speaker']}")
        print(f"🗣️ Text: {node_data['text']}")
        print(f"📌 Fact Summary: {node_data['fact_summary']}")
        print(f"🏷️ Topics: {node_data['topics']}")
        print(f"🧩 Details: {node_data['details']}")
        print(f"🎯 Confidence: {node_data['confidence']}")
        print("-" * 60)
    return nodes, embeddings

def construct_continuation_edges(enriched_nodes):
    print("\n=============== Constructing Edges ===============")
    edges = []
    for i in range(len(enriched_nodes) - 1):
        curr = enriched_nodes[i]
        next_node = enriched_nodes[i + 1]
        if curr["speaker"] == next_node["speaker"]:
            edge_cont = { "from": curr["id"], "to": next_node["id"], "type": "continuation" }
            edges.append(edge_cont)
            print(f"⏩ Edge (Continuation): {curr['id']} → {next_node['id']} (Same speaker)")
    return edges

# Question & Answer Edges
def is_question(text):
    return "?" in text.strip()

def construct_qa_edges(nodes):
    print("\n=============== Question & Answer Edges ===============")
    edges = []
    for i, curr in enumerate(nodes):
        curr_id = curr["id"]
        curr_text = curr["text"]
        if is_question(curr_text):
            print(f"❓ Question Detected: {curr_id} | {curr_text}")
            for j in range(i + 1, len(nodes)):
                responder = nodes[j]
                responder_id = responder["id"]
                if responder_id != curr_id:
                    question_edge = {
                        "from": curr_id,
                        "to": responder_id,
                        "type": "question"
                    }
                    answer_edge = {
                        "from": responder_id,
                        "to": curr_id,
                        "type": "answer"
                    }
                    edges.extend([question_edge, answer_edge])
                    print(f"🧭 Question Edge: {question_edge['from']} → {question_edge['to']}")
                    print(f"💬 Answer Edge: {answer_edge['from']} → {answer_edge['to']}")
                    break  # Only first match used
    return edges

def construct_related_topic_edges(nodes, embeddings, min_similarity=0.50):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n=============== Pairwise Semantic Similarity Edges ===============")
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = cosine_similarity(
                np.array(embeddings[i]).reshape(1, -1),
                np.array(embeddings[j]).reshape(1, -1)
            )[0][0]

            shared = set(nodes[i]["topics"]) & set(nodes[j]["topics"])
            if sim >= min_similarity:
                edge = {
                    "from": nodes[i]["id"],
                    "to": nodes[j]["id"],
                    "type": "related_topic",
                    "similarity": round(float(sim), 4),
                    "shared_topics": list(shared)
                }
                edges.append(edge)
                print(f"🔗 Edge (Related Topic): {edge['from']} → {edge['to']} | Similarity: {edge['similarity']}")
    return edges
            
def construct_graph(enriched_nodes, embed_model, min_similarity=0.50):
    nodes, embeddings = construct_nodes(enriched_nodes, embed_model)
    edges = []
    edges += construct_continuation_edges(enriched_nodes)
    edges += construct_qa_edges(nodes)
    edges += construct_related_topic_edges(nodes, embeddings, min_similarity=min_similarity)

    print("\n=============== Final Edge List ===============")
    for edge in edges:
        print(edge)
    print("\n=============== Final Node List ===============")
    for node in nodes:
        print(node)

    return nodes, edges
    
#---------------GRAPH STORAGE, POPULATE GRAPH WITH DATA--------------
    
def build_graph(nodes, edges):
    """
    Creates a directed graph from node and edge dictionaries.
    """
    G = nx.DiGraph()

    # Add all nodes with metadata
    for node in nodes:
        G.add_node(node["id"], **node)

    # Add edges with attributes
    for edge in edges:
        G.add_edge(edge["from"], edge["to"], **{k: v for k, v in edge.items() if k not in ["from", "to"]})

    return G

def draw_graph(G, layout=None, with_labels=True):
    """
    Optional visualization helper using matplotlib.
    """

    if layout is None:
        layout = nx.shell_layout(G)

    edge_labels = nx.get_edge_attributes(G, 'type')
    node_labels = {node: node for node in G.nodes()} # Shows node IDs like n1, n2, etc. instead of speakers
    
    # Assign colors based on speaker
    colors = []
    for node in G.nodes(data=True):
        speaker = node[1].get("speaker", "")
        if speaker == "Telemarketer":
            colors.append("skyblue")
        elif speaker == "Customer":
            colors.append("lightgreen")
        else:
            colors.append("gray")

    nx.draw(G, layout, with_labels=with_labels, labels=node_labels, node_color=colors, node_size=1500, font_size=10, edge_color='gray')
    nx.draw_networkx_edge_labels(G, layout, edge_labels=edge_labels, font_color='red')

    plt.title("Conversation Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# #------------------------------Audit Criterion Stage 2.2------------------------------
# # Seperate function created to allow previous rule-based functions focused on atomic tagging
# # This function accesses other nodes rather than only one node
# def evaluate_stage2_2(enriched_chunks: List[Dict]) -> Dict[str, any]:
#     """
#     Stage 2.2: If the customer shows uncertainty,
#     check if the next telemarketer chunk proposes a meeting.
#     """
#     passed = False
#     justifications = []

#     for i, chunk in enumerate(enriched_chunks[:-1]):
#         curr = chunk
#         next_chunk = enriched_chunks[i + 1]

#         # Check if Customer shows uncertainty
#         if curr["speaker"].lower() == "customer" and "Customer uncertainty" in curr["fact_summary"]:
#             # Look ahead for TM proposing meeting
#             if next_chunk["speaker"].lower() == "telemarketer" and "Proposed meeting" in next_chunk["fact_summary"]:
#                 passed = True
#                 justifications.append({
#                     "customer_chunk": curr["text"],
#                     "tm_chunk": next_chunk["text"],
#                     "customer_index": i,
#                     "tm_index": i + 1
#                 })

#     return {
#         "criterion": "Stage 2.2 - Customer uncertainty followed by proposed meeting",
#         "passed": passed,
#         "justifications": justifications
#     }

#-------------------Subgraph Retrieval for each audit criteria------------------- 
def retrieve_subgraph_for_criterion_fixed(
    criterion: str,
    nodes: list,
    edges: list,
    embed_model,
    min_score: float = 0.3,
    top_k: int = 3
) -> dict:
    print(f"\n🔍 Retrieving Subgraph for Criterion: {criterion}")
    criterion_embedding = embed_model.get_text_embedding(criterion)
    scored_nodes = []

    for node in nodes:
        scores_sources = []

        try:
            summary_score = cosine_similarity(
                [criterion_embedding],
                [embed_model.get_text_embedding(node.get("fact_summary", ""))]
            )[0][0]
            scores_sources.append((summary_score, "fact_summary"))
        except:
            pass

        try:
            text_score = cosine_similarity(
                [criterion_embedding],
                [embed_model.get_text_embedding(node.get("text", ""))]
            )[0][0]
            scores_sources.append((text_score, "text"))
        except:
            pass

        for topic in node.get("topics", []):
            try:
                topic_score = cosine_similarity(
                    [criterion_embedding],
                    [embed_model.get_text_embedding(topic)]
                )[0][0]
                scores_sources.append((topic_score, f"topic: {topic}"))
            except:
                continue

        for detail in node.get("details", []):
            try:
                detail_score = cosine_similarity(
                    [criterion_embedding],
                    [embed_model.get_text_embedding(detail)]
                )[0][0]
                scores_sources.append((detail_score, f"detail: {detail}"))
            except:
                continue

        if scores_sources:
            max_score, match_source = max(scores_sources, key=lambda x: x[0])
            if max_score >= min_score:
                scored_nodes.append((node, max_score, match_source))

    def get_node_text(nid):
        return next((n["text"] for n in nodes if n["id"] == nid), "[Text not found]")

    # Sort and get top nodes
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [item[0] for item in scored_nodes[:top_k]]
    top_node_ids = set(n["id"] for n in top_nodes)

    # Collect extended nodes and edges
    all_included_node_ids = set(top_node_ids)
    final_edges = []

    for node in top_nodes:
        node_id = node["id"]

        # CONTINUATION edge
        continuation_edge = next(
            (e for e in edges if e["from"] == node_id and e["type"] == "continuation"), None)
        if continuation_edge:
            cont_node_id = continuation_edge["to"]
            all_included_node_ids.add(cont_node_id)
            final_edges.append(continuation_edge)

        # QUESTION/ANSWER edges
        # Q/A logic: prioritize question → answer or answer → question pairing
        qa_edge = None
        qa_connected_id = None

        # Check if this node is asking a question
        qa_edge = next((e for e in edges if e["type"] == "question" and e["from"] == node_id), None)
        if qa_edge:
            qa_connected_id = qa_edge["to"]
        else:
            # Check if this node is answering a question
            qa_edge = next((e for e in edges if e["type"] == "answer" and e["to"] == node_id), None)
            if qa_edge:
                qa_connected_id = qa_edge["from"]

        if qa_edge and qa_connected_id:
            all_included_node_ids.add(qa_connected_id)
            final_edges.append(qa_edge)
            qa_text = get_node_text(qa_connected_id)
        else:
            # RELATED TOPIC fallback
            related_edges = [
                e for e in edges
                if e["type"] == "related_topic" and e["from"] == node_id
            ]
            if related_edges:
                best_related = max(related_edges, key=lambda e: e.get("similarity", 0))
                all_included_node_ids.add(best_related["to"])
                final_edges.append(best_related)

    # Filter nodes to return only those involved
    final_nodes = [node for node in nodes if node["id"] in all_included_node_ids]

    # Logging
    for node, score, source in scored_nodes[:top_k]:
        print(f"\n🟩 Node {node['id']} (Max Score: {round(score, 3)})")
        print(f"📌 Best match from: {source}")
        print(f"👤 Speaker: {node['speaker']}")
        print(f"🗣️ Text: {node['text']}")
        print(f"📌 Summary: {node['fact_summary']}")
        print(f"🏷️ Topics: {node['topics']}")
        print(f"🧩 Details: {node['details']}")
        print(f"🎯 Confidence: {node['confidence']}")

        # Connected nodes log
        node_id = node["id"]
        print(f"🟦 Connected nodes for {node_id}:")
        
        cont_edge = next((e for e in edges if e["from"] == node_id and e["type"] == "continuation"), None)

        # Continuation
        if cont_edge:
            cont_text = get_node_text(cont_edge["to"])
            print(f"   - Continuation ➡️ {cont_edge['to']}: {cont_text}")

        # Q/A (recalculate within log scope)
        qa_edge_log = next((e for e in edges if e["type"] == "question" and e["from"] == node_id), None)
        if qa_edge_log:
            qa_connected_id = qa_edge_log["to"]
            qa_text = get_node_text(qa_connected_id)
            print(f"   - Q/A ➡️ {qa_connected_id}: {qa_text}")
        else:
            qa_edge_log = next((e for e in edges if e["type"] == "answer" and e["to"] == node_id), None)
            if qa_edge_log:
                qa_connected_id = qa_edge_log["from"]
                qa_text = get_node_text(qa_connected_id)
                print(f"   - Q/A ➡️ {qa_connected_id}: {qa_text}")
            else:
                # fallback
                related_edges = [
                    e for e in edges if e["type"] == "related_topic" and e["from"] == node_id
                ]
                if related_edges:
                    best_related = max(related_edges, key=lambda e: e.get("similarity", 0))
                    rel_text = get_node_text(best_related["to"])
                    print(f"   - Related Topic ➡️ {best_related['to']} (similarity: {round(best_related['similarity'], 3)}): {rel_text}")

    return {
        "criterion": criterion,
        "top_nodes": final_nodes,
        "edges": final_edges,
        "log": [(n, float(score), source) for (n, score, source) in scored_nodes[:top_k]]
    }
  


#-------------------COMBINING OUTPUTS OF VECTORRAG AND GRAPHRAG------------------- 

# Function to combine and print results for each criterion
def combine_rag_outputs(vector_rag: Dict[str, List[Tuple[int, float, str]]],
                        graph_rag: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("------------------COMBINING OUTPUTS OF VECTORRAG AND GRAPHRAG----------------")
    combined_output = []
    
    for entry in graph_rag:
        criterion = normalize_criterion(entry["criterion"])
        graph_nodes = entry["top_nodes"]
        vector_chunks = vector_rag.get(criterion, [])
        
        combined_output.append({
            "criterion": criterion,
            "vector_chunks": [
                {
                    "chunk_id": int(chunk[0]),
                    "score": float(chunk[1]),
                    "text": chunk[2]
                } for chunk in vector_chunks
            ],
            "graph_nodes": [
                {
                    "node_id": node["id"],
                    "speaker": node["speaker"],
                    "text": node["text"],
                    "confidence": node["confidence"]
                } for node in graph_nodes
            ]
        })
    
    return combined_output
        


def print_combined_results(results):
    print("=" * 25 + " COMBINED RESULTS " + "=" * 25)
    
    for idx, result in enumerate(results):
        print(f"\n[{idx+1}] Criterion:")
        print(f"  {result['criterion']}\n")
        
        # Print Vector Chunks
        vector_chunks = result.get("vector_chunks", [])
        if vector_chunks:
            print("  Vector Chunks:")
            for chunk in vector_chunks:
                print(f"    - Chunk ID: {chunk['chunk_id']}")
                print(f"      Score: {chunk['score']:.3f}")
                print(f"      Text: {chunk['text'].strip()[:200]}{'...' if len(chunk['text']) > 200 else ''}\n")
        else:
            print("  Vector Chunks: None\n")

        # Print Graph Nodes
        graph_nodes = result.get("graph_nodes", [])
        if graph_nodes:
            print("  Graph Nodes:")
            for node in graph_nodes:
                print(f"    - Node ID: {node['node_id']}")
                print(f"      Speaker: {node['speaker']}")
                print(f"      Confidence: {node['confidence']:.3f}")
                print(f"      Text: {node['text'].strip()[:200]}{'...' if len(node['text']) > 200 else ''}\n")
        else:
            print("  Graph Nodes: None\n")
    
    print("=" * 70)


# ------------------------------AUDITING RESULTS----------------------------------

def combined_audit(combined_result: list[dict], model_engine="gpt-4o-mini", stage=1):
    print("==================AUDITING RESULTS:===============\n")
    client = llm
    output_results = []
    overall_result = "Pass"
    fail_count = 0

    for entry in combined_result:
        criterion = entry["criterion"]
        vector_chunks = entry.get("vector_chunks", [])
        graph_nodes = entry.get("graph_nodes", [])

        # Combine all retrieved text into one transcript
        vector_text = "\n\n".join(chunk["text"] for chunk in vector_chunks)
        graph_text = "\n\n".join(node["text"] for node in graph_nodes)
        combined_text = (vector_text + "\n\n" + graph_text).strip()

        print(f"\n--- Auditing Criterion ---\n{criterion}\n")

        prompt = f"""
        You are an auditor for IPP or IPPFA. 
        You are tasked with auditing a conversation between a telemarketer from IPP or IPPFA and a customer. 
        The audit evaluates whether the telemarketer adhered to a specific criterion from a predefined list.

        ### Instruction:
        - Review the provided conversation transcript.
        - Assess the telemarketer's compliance **only for the following single criterion**:
            "{criterion}"
        - Quote specific reasons from the conversation to justify the result.
        - Only mark a criterion as "Pass" if you are very confident (i.e., nearly certain) based on clear and specific evidence.
        - If not applicable, you may return "Not Applicable".

        ### Input Transcript:
        {combined_text}

        ### Response Format (JSON):
        [
            {{
                "Criteria": "{criterion}",
                "Reason": "<Your explanation>",
                "Result": "Pass" or "Fail" or "Not Applicable"
            }}
        ]
        """

        response = client.chat.completions.create(
            model=model_engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        result_text = response.choices[0].message.content
        cleaned = (
            result_text.replace("```json", "")
            .replace("```", "")
            .replace("### Response:", "")
            .strip()
        )

        try:
            result_json = json.loads(cleaned)
        except Exception as e:
            print(f"[!] JSON parsing error for criterion:\n{criterion}")
            print("Raw output:\n", result_text)
            raise e

        output_results.append(result_json[0])
        if result_json[0]["Result"] == "Fail":
            fail_count += 1

    # Overall decision
    if (stage == 1 and fail_count > 2) or (stage == 2 and fail_count > 1):
        overall_result = "Fail"

    final_output = {
        f"Stage {stage}": output_results,
        "Overall Result": overall_result
    }

    return final_output

# ========================================Stage 1============================================
# ========================================Stage 1============================================
# ========================================Stage 1============================================

# Splitting sentences and chunking
raw_chunks = preprocess_transcript(doc)
print_preprocessed_transcript(raw_chunks)

# Metadata construction for each sentence
enriched_nodes = enrich_chunks(raw_chunks, llmTF=True)
print("\n=============== Enriched Transcript Nodes ===============")
print_enriched_nodes(enriched_nodes)

# Node & Edge Construction
nodes, edges = construct_graph(enriched_nodes, embed_model, min_similarity=0.50)

# Knowledge Graph Construction
G = build_graph(nodes, edges)
draw_graph(G)

stage_1_criteria = [
    "Did the telemarketer introduced themselves by stating their name? (Usually followed by 'calling from')",
    "Did the telemarketer state that they are calling from one of these ['IPP', 'IPPFA', 'IPP Financial Advisors'] without mentioning on behalf of any other insurers?(accept anyone one of the 3 name given)",
    "Did the customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)",
    "Did the telemarketer specify the types of financial services offered?",
    "Did the telemarketer offered to set up a meeting or zoom session with the consultant for the customer? (Try to specify the date and location if possible)",
    "Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)",
    "Was the telemarketer polite and professional in their conduct?"
]

# Run for all Stage 1 criteria
stage1_subgraphs = []

for criterion in stage_1_criteria:
    result = retrieve_subgraph_for_criterion_fixed(criterion, nodes, edges, embed_model, min_score=0)
    stage1_subgraphs.append(result)

# Example: print top node IDs for first criterion
print("\n✅ Top Node IDs for First Criterion:")
for node in stage1_subgraphs[0]["top_nodes"]:
    print(f"- {node['id']}")

print(stage1_subgraphs)

# CLEANING VECTOR RAG OUTPUT AND TURNING INTO DICTIONARY

graph_rag = stage1_subgraphs
vector_rag = r'''
{"Did the telemarketer introduced themselves by stating their name? (Usually followed by 'calling from')": [(np.int64(0), 0.5699628470235988, 'Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she 
sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nTelemarketer: This is regarding what?'), (np.int64(4), 0.483634983610231, 'Telemarketer: Alright, no worries. Thank you.\nCustomer: Thanks.\nTelemarketer: Thank you.'), (np.int64(3), 0.3810681442556919, 'Customer: Should be ok, yeah.\nTelemarketer: Should be ok, alright. What I can do is tentatively set a meeting for 12 p.m. on 9th March and closer to that date, maybe I can drop you a follow-up.\nCustomer: Sure, sure, sure.\nTelemarketer: Ok, thanks. And your email, maybe I can send you a Zoom invite via the email as well.\nCustomer: 
Yeah, you can send me over the text, right? This is a WhatsApp number, you can send me over there on WhatsApp.\nTelemarketer: Alright, no worries. Thank you.')], "Did the telemarketer state that they are calling from one of 
these ['IPP', 'IPPFA', 'IPP Financial Advisors'] without mentioning on behalf of any other insurers?(accept anyone one of the 3 name given)": [(np.int64(0), 0.5779707128892415, 'Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nTelemarketer: This is regarding what?'), (np.int64(1), 0.5094205307915771, "Telemarketer: This is regarding what?\nTelemarketer: Yeah, so very quickly, I'm from IPP Financial Advisors, so basically we are Singapore's largest and oldest financial advisory firm, and we help expats and professionals specifically, like yourself, with everything from offshore investments, retirement planning, university fee planning, insurance, offshore investments as well.\nTelemarketer: So just wanted to check with you 
if you might be keen to, you know, jump on a 15 to 20 minute Zoom session, just a sharing session with yourself 
and Elena Pryor, who's our senior consultant, maybe next week or the week after?\nCustomer: Yeah, I'm not interested right now in making any investments, so that's the reason maybe I have not responded.\nTelemarketer: I see.\nTelemarketer: That's ok. Maybe if you just want to learn more and we can exchange business cards as well, that way we can stay in touch whenever you might be more keen and more interested to."), (np.int64(4), 0.4166529257349394, 'Telemarketer: Alright, no worries. Thank you.\nCustomer: Thanks.\nTelemarketer: Thank you.')], "Did the 
customer asked how did the telemarketer obtained their contact details? If they asked, did telemarketer mentioned who gave the customer's details to him? (Not Applicable if customer didn't)": [(np.int64(0), 0.5338795942954225, 'Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nTelemarketer: This is regarding what?'), (np.int64(4), 0.4157951889440595, 'Telemarketer: Alright, no worries. Thank you.\nCustomer: Thanks.\nTelemarketer: Thank you.'), (np.int64(3), 0.40944460595740406, 'Customer: Should be ok, yeah.\nTelemarketer: Should be ok, alright. What I can do is tentatively set a meeting for 12 p.m. on 9th March and closer to that date, maybe I can drop you a follow-up.\nCustomer: Sure, sure, sure.\nTelemarketer: Ok, thanks. And your email, maybe I can send you a Zoom invite via the email as well.\nCustomer: Yeah, you can send me over the text, right? This is a WhatsApp number, you can send me over there on WhatsApp.\nTelemarketer: Alright, no worries. Thank you.')], 'Did the telemarketer specify the types of financial services offered?': [(np.int64(1), 0.4745612899258858, "Telemarketer: This is regarding what?\nTelemarketer: Yeah, so very quickly, I'm from IPP Financial Advisors, so basically we are Singapore's largest and oldest financial advisory firm, and we help expats and professionals specifically, like yourself, with everything from offshore investments, retirement planning, university fee planning, insurance, offshore investments as well.\nTelemarketer: So just wanted to check with you if you might be keen to, you know, jump on a 15 to 20 minute Zoom session, just a sharing session with yourself and Elena Pryor, who's our senior consultant, maybe next week or the week after?\nCustomer: Yeah, I'm not interested right now in making any investments, so that's the reason maybe I have not responded.\nTelemarketer: I see.\nTelemarketer: That's ok. Maybe if you just want to learn more and we can exchange business cards as well, that way we can stay in touch whenever you might be more keen and more interested to."), (np.int64(4), 0.4182199878248002, 'Telemarketer: Alright, no worries. Thank you.\nCustomer: Thanks.\nTelemarketer: Thank you.'), (np.int64(0), 0.4154893295605243, 'Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nTelemarketer: This is regarding what?')], 'Did the telemarketer offered to set up a meeting or 
zoom session with the consultant for the customer? (Try to specify the date and location if possible)': [(np.int64(3), 0.5789961316063704, 'Customer: Should be ok, yeah.\nTelemarketer: Should be ok, alright. What I can do is tentatively set a meeting for 12 p.m. on 9th March and closer to that date, maybe I can drop you a follow-up.\nCustomer: Sure, sure, sure.\nTelemarketer: Ok, thanks. And your email, maybe I can send you a Zoom invite via the email as well.\nCustomer: Yeah, you can send me over the text, right? This is a WhatsApp number, you can send 
me over there on WhatsApp.\nTelemarketer: Alright, no worries. Thank you.'), (np.int64(0), 0.5326632597669154, 'Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, 
sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nTelemarketer: This is regarding what?'), (np.int64(1), 0.4745825804851915, "Telemarketer: This is regarding what?\nTelemarketer: Yeah, so very quickly, I'm from IPP Financial Advisors, so basically we are Singapore's largest and oldest financial advisory firm, and we help expats and professionals specifically, like yourself, with everything from offshore investments, retirement planning, university fee planning, insurance, offshore investments as well.\nTelemarketer: So just wanted to check with you if you might be keen to, you know, jump on a 15 to 20 minute Zoom 
session, just a sharing session with yourself and Elena Pryor, who's our senior consultant, maybe next week or the week after?\nCustomer: Yeah, I'm not interested right now in making any investments, so that's the reason maybe I have not responded.\nTelemarketer: I see.\nTelemarketer: That's ok. Maybe if you just want to learn more and we can exchange business cards as well, that way we can stay in touch whenever you might be more keen and more interested to.")], "Did the telemarketer stated that products have high returns, guaranteed returns, or capital guarantee? (Fail if they did, Pass if they didn't)": [(np.int64(4), 0.4033207039223197, 'Telemarketer: Alright, no worries. Thank you.\nCustomer: Thanks.\nTelemarketer: Thank you.'), (np.int64(0), 0.39696216182587585, 'Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nTelemarketer: This is regarding what?'), (np.int64(1), 0.3698426786768071, "Telemarketer: This is regarding what?\nTelemarketer: Yeah, so very quickly, I'm from IPP Financial Advisors, so basically we are Singapore's largest and oldest financial advisory firm, and we help expats and professionals specifically, like yourself, with everything from offshore investments, retirement planning, university fee planning, insurance, offshore investments as well.\nTelemarketer: So just wanted to check with you if you might be keen to, you know, jump on a 15 to 20 minute Zoom session, just a sharing session with yourself and Elena Pryor, who's our senior consultant, maybe next week or the week after?\nCustomer: Yeah, I'm not interested right now in making any investments, so that's the reason maybe I have not responded.\nTelemarketer: I see.\nTelemarketer: That's ok. Maybe if you just want to learn more and we can exchange business cards as well, that way we can stay in touch whenever you might be more keen and more interested to.")], 'Was the telemarketer polite and professional in their conduct?': [(np.int64(0), 0.5789749716156185, 'Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: 
Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nTelemarketer: This is regarding what?'), (np.int64(4), 0.5581267969296707, 'Telemarketer: Alright, no worries. Thank 
you.\nCustomer: Thanks.\nTelemarketer: Thank you.'), (np.int64(3), 0.4727777632710083, 'Customer: Should be ok, 
yeah.\nTelemarketer: Should be ok, alright. What I can do is tentatively set a meeting for 12 p.m. on 9th March 
and closer to that date, maybe I can drop you a follow-up.\nCustomer: Sure, sure, sure.\nTelemarketer: Ok, thanks. And your email, maybe I can send you a Zoom invite via the email as well.\nCustomer: Yeah, you can send me over the text, right? This is a WhatsApp number, you can send me over there on WhatsApp.\nTelemarketer: Alright, no worries. Thank you.')]}
'''.strip()
# Step 1: Replace np.int64(x) with just x using regex
cleaned = re.sub(r'np\.int64\((\d+)\)', r'\1', vector_rag)
# Step 3: Escape line breaks inside string literals
cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
# Step 3: Strip leading newlines/spaces
cleaned = cleaned.strip()
print("CLEANED:", cleaned)
# Step 2: Use literal_eval (safe version of eval) to convert to dict
vector_rag_dict = ast.literal_eval(cleaned)
print("TYPEE:", type(vector_rag_dict))

# To ensure that criterion does not contain any newlines and matches between graph and vector jsons
def normalize_criterion(text: str) -> str:
    return " ".join(text.split()).strip()

vector_rag_dict = {
    normalize_criterion(k): v for k, v in vector_rag_dict.items()
}

# ====================Combine and prepare for display========================
combined_result = combine_rag_outputs(vector_rag_dict, graph_rag)

# =========================RESULTS=========================
print_combined_results(combined_result)

# =========================AUDITING========================
print(combined_audit(combined_result, stage=1))

# ================================Stage 2====================================
# ================================Stage 2====================================
# ================================Stage 2====================================

print("STAGE 2 STARTING PROCESSING")

stage_2_criteria = [
    "Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?",
    "Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?",
    "Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)"
]

stage2_subgraphs = []

for criterion in stage_2_criteria:
    result = retrieve_subgraph_for_criterion_fixed(criterion, nodes, edges, embed_model, min_score=0)
    stage2_subgraphs.append(result)

print(stage2_subgraphs)

print("------------------COMBINING OUTPUTS OF STAGE 2 VECTORRAG AND GRAPHRAG----------------")

graph_rag = stage2_subgraphs
vector_rag = r'''
{
  "Did the telemarketer ask if the customer is keen to explore how they can benefit from IPPFA's services?": [
    [0, 0.5279, "Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nCustomer: This is regarding what?"],
    [1, 0.4339, "Customer: This is regarding what?\nTelemarketer: Yeah, so very quickly, I'm from IPP Financial Advisors, so basically we are Singapore's largest and oldest financial advisory firm, and we help expats and professionals specifically, like yourself, with everything from offshore investments, retirement planning, university fee planning, insurance, offshore investments as well.\nTelemarketer: So just wanted to check with you if you might be keen to, you know, jump on a 15 to 20 minute Zoom session, just a sharing session with yourself and Elena Pryor, who's our senior consultant, maybe next week or the week after.\nCustomer: Yeah, I'm not interested right now in making any investments, so that's the reason maybe I have not responded.\nTelemarketer: I see.\nTelemarketer: That's ok. Maybe if you just want to learn more and we can exchange business cards as well, that way we can stay in touch whenever you might be more keen and more interested to."],
    [4, 0.3989, "Telemarketer: Alright, no worries. Thank you.\nCustomer: Thanks.\nTelemarketer: Thank you."]
  ],
  "Did the customer show uncertain response to the offer of the product and services? If Yes, Check did the telemarketer propose meeting or zoom session with company's consultant?": [
    [0, 0.5165, "Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nCustomer: This is regarding what?"],
    [4, 0.4971, "Telemarketer: Alright, no worries. Thank you.\nCustomer: Thanks.\nTelemarketer: Thank you."],
    [3, 0.4914, "Customer: Should be ok, yeah.\nTelemarketer: Should be ok, alright. What I can do is tentatively set a meeting for 12 p.m. on 9th March and closer to that date, maybe I can drop you a follow-up.\nCustomer: Sure, sure, sure.\nTelemarketer: Ok, thanks. And your email, maybe I can send you a Zoom invite via the email as well.\nCustomer: Yeah, you can send me over the text, right? This is a WhatsApp number. You can send me over here on WhatsApp.\nTelemarketer: Alright, no worries. Thank you."]
  ],
  "Did the telemarketer pressure the customer for the following activities (product introduction, setting an appointment)? (Fail if they did, Pass if they didn't)": [
    [0, 0.4779, "Telemarketer: Hello Hi Amit, this is Mihir calling from IPP. Are you ok to talk at the moment?\nCustomer: Yeah, sure.\nTelemarketer: Yes, I was passed on your details from my senior consultant, Elena Pryor. Does that name ring a bell to you?\nCustomer: Elena, no.\nTelemarketer: Ok, so no worries. So she sent a couple of messages on LinkedIn a while ago to set up a meeting, but never heard back from you, so just wanted to follow up.\nCustomer: This is regarding what?"],
    [4, 0.4304, "Telemarketer: Alright, no worries. Thank you.\nCustomer: Thanks.\nTelemarketer: Thank you."],
    [3, 0.4198, "Customer: Should be ok, yeah.\nTelemarketer: Should be ok, alright. What I can do is tentatively set a meeting for 12 p.m. on 9th March and closer to that date, maybe I can drop you a follow-up.\nCustomer: Sure, sure, sure.\nTelemarketer: Ok, thanks. And your email, maybe I can send you a Zoom invite via the email as well.\nCustomer: Yeah, you can send me over the text, right? This is a WhatsApp number. You can send me over here on WhatsApp.\nTelemarketer: Alright, no worries. Thank you."]
  ]
}
'''.strip()
vector_rag_dict = json.loads(vector_rag)
# Step 1: Replace np.int64(x) with just x using regex
cleaned = re.sub(r'np\.int64\((\d+)\)', r'\1', vector_rag)
# Step 3: Escape line breaks inside string literals
cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
# Step 3: Strip leading newlines/spaces
cleaned = cleaned.strip()
print("VECTOR_RAG_DICT:", vector_rag_dict)
print("TYPE:", type(vector_rag_dict))
print("CLEANED:", cleaned)

# To ensure that criterion does not contain any newlines and matches between graph and vector jsons
def normalize_criterion(text: str) -> str:
    return " ".join(text.split()).strip()

vector_rag_dict = {
    normalize_criterion(k): v for k, v in vector_rag_dict.items()
}

# Combine and prepare for display
combined_result = combine_rag_outputs(vector_rag_dict, graph_rag)

print("==================COMBINED RESULTS:===============\n")

print_combined_results(combined_result)

print("==================AUDITING RESULTS:===============\n")

print(combined_audit(combined_result, stage=2))