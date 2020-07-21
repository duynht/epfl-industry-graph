python3 async_extract_entities.py \
--datapath ../data/

# usage: async_extract_entities.py [-h] [-n NUM_ENTRIES] [-d DATAPATH]

# optional arguments:
#   -h, --help            show this help message and exit
#   -n NUM_ENTRIES, --num_entries NUM_ENTRIES
#                         number of entries to be extracted (defaulted to
#                         extract all)
#   -d DATAPATH, --datapath DATAPATH
#                         path to data directory (default is "../data")

python3 graph_parser.py \
--datapath ../data/

# usage: graph_parser.py [-h] [-n NUM_NODES] [-d DATAPATH]

# optional arguments:
#   -h, --help            show this help message and exit
#   -n NUM_NODES, --num_nodes NUM_NODES
#                         number of nodes to be parsed (defaulted to parse all)
#   -d DATAPATH, --datapath DATAPATH
#                         path to data directory (default is "../data")

python3 Splitter/src/main.py \
--edge-path ../data/parsed-graph/pt_graph.csv \
--embedding-output-path ../data/embeddings/persona_embedding.csv \
--persona-output-path ../data/embeddings/pt_persona_map.json \
--dimensions 128

# usage: main.py [-h] [--edge-path [EDGE_PATH]]
#                [--embedding-output-path [EMBEDDING_OUTPUT_PATH]]
#                [--persona-output-path [PERSONA_OUTPUT_PATH]]
#                [--number-of-walks NUMBER_OF_WALKS] [--window-size WINDOW_SIZE]
#                [--negative-samples NEGATIVE_SAMPLES]
#                [--walk-length WALK_LENGTH] [--seed SEED]
#                [--learning-rate LEARNING_RATE] [--lambd LAMBD]
#                [--dimensions DIMENSIONS] [--workers WORKERS]

# Run Splitter.

# optional arguments:
#   -h, --help            show this help message and exit
#   --edge-path [EDGE_PATH]
#                         Edge list csv.
#   --embedding-output-path [EMBEDDING_OUTPUT_PATH]
#                         Embedding output path.
#   --persona-output-path [PERSONA_OUTPUT_PATH]
#                         Persona output path.
#   --number-of-walks NUMBER_OF_WALKS
#                         Number of random walks per source node. Default is 10.
#   --window-size WINDOW_SIZE
#                         Skip-gram window size. Default is 5.
#   --negative-samples NEGATIVE_SAMPLES
#                         Negative sample number. Default is 5.
#   --walk-length WALK_LENGTH
#                         Truncated random walk length. Default is 40.
#   --seed SEED           Random seed for PyTorch. Default is 42.
#   --learning-rate LEARNING_RATE
#                         Learning rate. Default is 0.025.
#   --lambd LAMBD         Regularization parameter. Default is 0.1.
#   --dimensions DIMENSIONS
#                         Embedding dimensions. Default is 128.
#   --workers WORKERS     Number of parallel workers. Default is 4.

cd web-ui

python3 pt_evaluation.py \
--datapath ../../data \
--result_path ../results

# usage: pt_evaluation.py [-h] [-d DATAPATH] [-rp RESULT_PATH] [-k TOP_K] [-gpu]

# optional arguments:
#   -h, --help            show this help message and exit
#   -d DATAPATH, --datapath DATAPATH
#                         path to data directory (default is "../../data")
#   -rp RESULT_PATH, --result_path RESULT_PATH
#                         path to result directory (default is "../results")
#   -k TOP_K, --top_k TOP_K
#                         top k nearest neighbors (default is 10)
#   -gpu, --use_gpu       use GPU for indexing (defaulted to not using)

python3 main.py \
--datapath ../../data \
--result_path ../results

# usage: main.py [-h] [-d DATAPATH] [-k TOP_K] [-gpu]

# optional arguments:
#   -h, --help            show this help message and exit
#   -d DATAPATH, --datapath DATAPATH
#                         path to data directory (default is "../../data")
#   -k TOP_K, --top_k TOP_K
#                         Top k nearest neighbors (default is 10)
#   -gpu, --use_gpu       Use GPU for indexing (defaulted to not using)

