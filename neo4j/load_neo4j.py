from py2neo import Database, Graph, Node, Relationship
import os
import argparse
import pandas as pd

def load_companies(filepath, graph):
    query = """
    USING PERIODIC COMMIT 1000
    LOAD CSV WITH HEADERS FROM "file:///{filepath}" AS row
    MERGE (company:Company {{uid: row.uid, normalized_name: row.normalized_name, name: row.name}})
    """.format(filepath=filepath)

    graph.run(query)


def load_fields(filepath, graph):
    query = """
    USING PERIODIC COMMIT 1000
    LOAD CSV WITH HEADERS FROM "file:///{filepath}" AS row
    MERGE (field:Field {{wikidataId: row.wikidataId, pageId: row.pageId, label: row.label}})
    """.format(filepath=filepath)

    graph.run(query)

def load_relationships(filepath, graph):
    query = """
    USING PERIODIC COMMIT 1000
    LOAD CSV WITH HEADERS FROM "file:///{filepath}" AS row
    MATCH (company:Company {{uid: row.company}})
    MATCH (field:Field {{wikidataId: row.field}})
    MERGE (company)-[:WORKS_ON]->(field)
    """.format(filepath=filepath)

    graph.run(query)

def prune_unknown(graph):
    query = """
    MATCH (field:Field {{label: "Unknown"}})
    DETACH DELETE field
    """

    graph.run(query)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', 
                        default='../../data',
                        type=str,
                        help='path to data directory (default is "../data")')

    args = parser.parse_args()
    graph = Graph('bolt://localhost:7687')
    graph.delete_all()
    
    load_companies(os.path.join(args.datapath, 'companies.csv'), graph)
    load_fields(os.path.join(args.datapath, 'fields.csv'), graph)
    load_relationships(os.path.join(args.datapath, 'relationships.csv'), graph)
    prune_unknown(graph)