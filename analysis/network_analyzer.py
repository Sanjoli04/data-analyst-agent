# analysis/network_analyzer.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

def perform_network_analysis(file_paths, analysis_requests):
    """
    Performs network analysis on an edge list file and returns a list of answers.
    """
    if not file_paths:
        return {"error": "No data file path provided for network analysis."}
    
    file_path = file_paths[0]
    
    try:
        df = pd.read_csv(file_path)
        G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])

        answers = []
        
        # --- Perform Requested Analyses & Format Answers ---
        for req in analysis_requests:
            req_type = req.get('type')
            
            if req_type == 'edge_count':
                answers.append(f"Number of edges in the network: {G.number_of_edges()}")

            elif req_type == 'highest_degree_node':
                degrees = dict(G.degree())
                if degrees:
                    highest_degree_node = max(degrees, key=degrees.get)
                    answers.append(f"Node with the highest degree: {highest_degree_node}")
                else:
                    answers.append("Node with the highest degree: N/A")

            elif req_type == 'average_degree':
                degrees = dict(G.degree())
                if degrees:
                    avg_degree = np.mean(list(degrees.values()))
                    answers.append(f"Average degree of the network: {round(avg_degree, 2)}")
                else:
                    answers.append("Average degree of the network: 0")

            elif req_type == 'density':
                answers.append(f"Network density: {round(nx.density(G), 4)}")

            elif req_type == 'shortest_path':
                source = req.get('params', {}).get('source')
                target = req.get('params', {}).get('target')
                try:
                    path_length = nx.shortest_path_length(G, source=source, target=target)
                    answers.append(f"Length of the shortest path between {source} and {target}: {path_length}")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    answers.append(f"No path found between {source} and {target}.")

            elif req_type == 'network_graph':
                plt.figure(figsize=(12, 12))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=10)
                plt.title("Network Graph")
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                answers.append(f"data:image/png;base64,{b64_str}")

            elif req_type == 'degree_histogram':
                degrees = [G.degree(n) for n in G.nodes()]
                plt.figure(figsize=(10, 6))
                plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), color='green', alpha=0.7, rwidth=0.8)
                plt.title("Degree Distribution")
                plt.xlabel("Degree")
                plt.ylabel("Frequency")
                plt.grid(axis='y', alpha=0.75)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                answers.append(f"data:image/png;base64,{b64_str}")

        return {"answers": answers}

    except Exception as e:
        return {"error": f"Failed to perform network analysis: {e}"}
