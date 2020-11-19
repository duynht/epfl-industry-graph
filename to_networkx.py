from twitter_swiss_actors import TwitterSwissActors as TSA
path = '../data/twitter_swiss_actors/'
dataset = TSA(path)
dataset.to_networkx('tsa_graph.json')