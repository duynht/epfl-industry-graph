from flask import Flask, render_template, url_for
from flask_ngrok import run_with_ngrok
from forms import ExplorerForm
from evaluate_neo4j import Neo4jEvaluator, NodeType, QueryType
from py2neo import Graph
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', 
                        default='../../data',
                        type=str,
                        help='path to data directory (default is "../../data")')
    
    parser.add_argument('-k', '--top_k', 
                        default=10,
                        type=int,
                        help='Top k nearest neighbors (default is 10)')

    parser.add_argument('-gpu', '--use_gpu',
                        default=False,
                        action='store_true',
                        help='Use GPU for indexing (defaulted to not using)')

    args = parser.parse_args()  

    app = Flask(__name__)
    app.config.update(dict(
        SECRET_KEY="powerful secretkey",
        WTF_CSRF_SECRET_KEY="a csrf secret key"
    ))
    run_with_ngrok(app)  # Start ngrok when app is run
    evaluator = Neo4jEvaluator(graph=Graph('bolt://localhost:7687'), datapath=args.datapath, top_k=args.top_k, use_gpu=args.use_gpu)

    eval_company2company = evaluator.get_eval_company2company()
    eval_company2field = evaluator.get_eval_company2field()
    eval_field2company = evaluator.get_eval_field2company()

    @app.route("/", methods=['GET','POST'])
    @app.route("/explorer", methods=['GET','POST'])
    def explorer():
        form = ExplorerForm()
        data = {}
        if form.query_type.data == QueryType['company2company']:
            form.node.choices = eval_company2company
        elif form.query_type.data == QueryType['company2field']:
            form.node.choices = eval_company2field
        else:
            form.node.choices = eval_field2company

        if form.validate_on_submit():
            # data = db.get_related_nodes(form.node_str.data, form.src_type.data, form.dest_type.data)
            # precision, recall, neighbors, ground_truth = evaluator.evaluate_node(ori_node_str=form.node_str.data, \
            #                                                                     src_type=NodeType[form.src_type.data], \
            #                                                                     dst_type=NodeType[form.dst_type.data], \
            #                                                                     zefix_uid=form.zefix_uid.data)
            precision, recall, neighbors, ground_truth = evaluator.evaluate_node(query_type=form.query_type.data, \
                                                                                node_id=form.node.data)
        
            data = {'precision' : precision, 'recall' : recall, 'neighbors' : neighbors, 'ground truth' : ground_truth}

            app.logger.info(str(precision))

        return render_template('explorer_form.html', title='Neighbor Explorer', form=form, data=data)

    app.run()
    # https://9e0faf39ab95.ngrok.io/
    # Garage Fuhrer AG CHE106659089
    # Google Switzerland GmbH CHE-110.474.423
    # Durena AG CHE106549207
    # Auguste Reymond S.A. CHE-100.135.008