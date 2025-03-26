import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create a Directed Graph
G = nx.DiGraph()
nodes = {
    "Audio Recording": "Raw customer-agent conversation",
    "Speech-to-Text": "Transcribed version of the conversation",
    "Transcript": "Structured text representation",
    "AI Audio Auditor": "System analyzing compliance",
    "Audit Criteria": "Rules for evaluating conversations",
    "Compliance Question": "Generated questions to assess adherence",
    "Agent": "Telemarketer conducting the call",
    "Customer": "Recipient of the call",
    "Evaluation Outcome": "Pass/Fail result of the audit",
    "Audit Report": "Final record of compliance assessment"
}
for node, desc in nodes.items():
    G.add_node(node, description=desc)

edges = [
    ("Audio Recording", "Speech-to-Text", "converted to"),
    ("Speech-to-Text", "Transcript", "forms"),
    ("Transcript", "AI Audio Auditor", "analyzed by"),
    ("AI Audio Auditor", "Audit Criteria", "compares against"),
    ("Audit Criteria", "Compliance Question", "derived from"),
    ("AI Audio Auditor", "Evaluation Outcome", "evaluates"),
    ("Agent", "Customer", "makes call to"),
    ("Customer", "Agent", "responds to"),
    ("Evaluation Outcome", "Audit Report", "stored in")
]
for src, dst, rel in edges:
    G.add_edge(src, dst, relation=rel)

# Step 2: Create embeddings for nodes and edges
model = SentenceTransformer('all-MiniLM-L6-v2')

# Node embeddings
node_embeddings = {node: model.encode(desc) for node, desc in nodes.items()}

# Edge embeddings
edge_embeddings = { 
    (src, dst): model.encode(f"{src} {rel} {dst}") 
    for src, dst, rel in edges
}

# Step 3: Retrieve based on query (Vector search)
def retrieve_from_kg_vector(query, node_embeddings, edge_embeddings, top_k=3):
    query_embedding = model.encode(query)
    
    node_similarities = cosine_similarity([query_embedding], list(node_embeddings.values()))[0]
    node_results = sorted(zip(node_embeddings.keys(), node_similarities), key=lambda x: x[1], reverse=True)[:top_k]
    
    edge_similarities = cosine_similarity([query_embedding], list(edge_embeddings.values()))[0]
    edge_results = sorted(zip(edge_embeddings.keys(), edge_similarities), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    
    for node, score in node_results:
        results.append(f"Entity: {node}, Description: {nodes[node]} (Similarity: {score:.3f})")
    
    for (src, dst), score in edge_results:
        # Directly access the relation from the edges list
        relation = next(rel for s, d, rel in edges if s == src and d == dst)
        results.append(f"Relation: {src} {relation} {dst} (Similarity: {score:.3f})")
    
    return results if results else ["No relevant knowledge found."]

# Example Query
query = "Did the telemarketer stated that products have high returns, guaranteed returns, or capital guaranteed?"
results = retrieve_from_kg_vector(query, node_embeddings, edge_embeddings)

# Output Results
print("📌 Retrieved Knowledge:")
for res in results:
    print(res)
