sudo apt -y install screen
sudo apt -y install libomp-dev

git config user.email "nhtduy@apcs.vn"
git config user.name "Tuan-Duy H. Nguyen"

pip3 install faiss faiss-gpu

mkdir data
scp -r -P 13492 tuanduy@0.tcp.ngrok.io:~/epfl-industry-graph/data/embeddings ./data/
scp -r -P 13492 tuanduy@0.tcp.ngrok.io:~/epfl-industry-graph/data/parsed-graph ./data/
scp -r -P 13492 tuanduy@0.tcp.ngrok.io:~/epfl-industry-graph/data/truth ./data/
