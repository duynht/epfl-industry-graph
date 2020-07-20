python3 extract_entities.py

python3 graph_parser.py 

python3 Splitter/src/main.py --edge-path ../data/parsed-graph/pt_graph.csv \
--embedding-output-path ../data/embeddings/persona_embedding.csv \
--persona-output-path ../data/embeddings/pt_persona_map.json \
--dimensions 128

cd web-ui

python pt_evaluation.py

python main.py

