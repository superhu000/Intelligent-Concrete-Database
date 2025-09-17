import json
import io
import math

from neo4j import GraphDatabase
from py2neo import Graph
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, FileResponse
from openpyxl import load_workbook
from .utils.dataExtrac import *
from .predict.predict import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# 连接Neo4j数据库
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "123456789"))
graph = Graph("bolt://localhost:7687", auth=("neo4j", "123456789"))


@csrf_exempt
def handle_excel_KG(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        try:
            df = pd.read_excel(uploaded_file, header=0, nrows=120)  # 使用pandas库读取Excel文件数据
            print(df)

            start_KG = 0
            end_KG = 100  # len(invoice_data)
            create_KG(df, start_KG, end_KG)

            df = df.drop(df.columns[0], axis=1)  # drop 第一列id 不显示在前端表格里
            print(df)

            feature = df.columns.tolist()
            not_empty_count = []
            for column in df.columns:
                not_empty_count.append(df[column].notnull().sum().item())
                # print(df[column].isnull().sum().item())

            max_value = df.max().tolist()
            max_value = [0 if math.isnan(x) or x is None else x for x in max_value]

            min_value = df.min().tolist()
            min_value = [0 if math.isnan(x) or x is None else x for x in min_value]

            average_value = df.mean().round(4).tolist()
            average_value = [0 if math.isnan(x) or x is None else x for x in average_value]

            # 构造结果数据
            result = {
                'feature': feature,
                'notEmptyCount': not_empty_count,
                'maxValue': max_value,
                'minValue': min_value,
                'averageValue': average_value
            }

            return JsonResponse({'success': True, 'result': result})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    else:
        return JsonResponse({'success': False, 'error': '未上传文件或请求方法不允许'})


@csrf_exempt
def search_create_KG(request):
    if request.method == "POST":
        data = json.loads(request.body)  # 前端向后端发送的参数  data = request.get_json(silent=True)
        n_node = data["nNodeList"]
        rel = data["relList"]
        m_node = data["mNodeList"]
        print(n_node)
        print(rel)
        print(m_node)

        dict1 = dataProcessing(n_node, rel, m_node)
        print(dict1)
        return JsonResponse(dict1, safe=False)


def dataProcessing(Nnode, Rel, Mnode):
    nodes = []
    links = []
    nodes_set = []

    n_node = ''
    rel = ''
    m_node = ''

    print("n_node: " + n_node + " rel: " + rel + " m_node: " + m_node)

    with driver.session() as session:
        result = session.run(
            'MATCH (n' + n_node + ')-[r' + rel + ']->(m' + m_node + ') RETURN ' +
            'id(n) as source, labels(n) as source_labels, properties(n) as source_properties, ' +
            'id(m) as target, labels(m) as target_labels, properties(m) as target_properties, ' +
            'id(r) as link, type(r) as r_type, properties(r) as r_properties ' +
            'LIMIT 100')

        for record in result:
            nodes.append({"id": record['source'], "label": record['source_labels'][0],
                          "properties": record['source_properties']})
            nodes.append({"id": record['target'], "label": record['target_labels'][0],
                          "properties": record['target_properties']})
            links.append({"source": record['source'], "target": record['target'], "type": record['r_type'],
                          "properties": record['r_properties']})

        for i in nodes:
            if i not in nodes_set:
                nodes_set.append(i)
        result_dict = dict(zip(['nodes', 'links'], [nodes_set, links]))
        # print(result_dict)
        return result_dict
