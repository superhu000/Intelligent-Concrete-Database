# -*- coding: utf-8 -*-
from py2neo import Node, Graph, Relationship, NodeMatcher, Subgraph
from concrete.utils.settings import *


class DataToNeo4j(object):
    """将excel中数据存入neo4j"""

    def __init__(self):
        """建立连接"""
        link = Graph(url, auth=(username, password))
        self.graph = link
        self.graph.delete_all()
        self.matcher = NodeMatcher(link)

    def create_node(self, data_list):
        """建立节点"""
        for data in data_list:
            for value in data["feature"]:
                node = Node(data["id"], name=value)
                # node.update(value: data["value"])
                self.graph.create(node)

    def create_node1(self, label, data_list):
        """建立节点with多个属性"""
        # for data in data_list:
        #     node = Node(label)
        #     for i in range(len(data_list[0])):
        #         node.update({property_list[i][0]:data[i]})
        for j in range(1, len(data_list)):
            node = Node(label)
            # node.update({"name": label+"编号"+data_list[j][0]})
            for i in range(len(data_list[0])):
                node.update({data_list[0][i][0]: data_list[j][i]})
            self.graph.create(node)

    def create_rel(self, df_data):
        """建立联系"""
        for m in range(0, len(df_data)):
            try:
                # print(list(self.matcher.match('C_CaO').where("_.name=" + "'" + df_data['C_CaO'][m] + "'")))
                rel = Relationship(
                    self.matcher.match('水掺量').where("_.name=" + "'" + df_data['水掺量'][m] + "'").first(),
                    "预测",
                    self.matcher.match('28d抗压强度').where("_.name=" + "'" + df_data['28d抗压强度'][m] + "'").first(),
                    name="预测"
                )
                rel1 = Relationship(
                    self.matcher.match('水灰比').where("_.name=" + "'" + df_data['水灰比'][m] + "'").first(),
                    "预测",
                    self.matcher.match('28d抗压强度').where("_.name=" + "'" + df_data['28d抗压强度'][m] + "'").first(),
                    name="预测"
                )
                rel2 = Relationship(
                    self.matcher.match('孔隙率').where("_.name=" + "'" + df_data['孔隙率'][m] + "'").first(),
                    "预测",
                    self.matcher.match('28d抗压强度').where("_.name=" + "'" + df_data['28d抗压强度'][m] + "'").first(),
                    name="预测"
                )
                rel3 = Relationship(
                    self.matcher.match('水泥').where("_.name=" + "'" + df_data['name'][m] + "'").first(),
                    "预测",
                    self.matcher.match('28d抗压强度').where("_.name=" + "'" + df_data['28d抗压强度'][m] + "'").first(),
                    name="预测"
                )
                rel4 = Relationship(
                    self.matcher.match('粉煤灰').where("_.name=" + "'" + df_data['name'][m] + "'").first(),
                    "预测",
                    self.matcher.match('28d抗压强度').where("_.name=" + "'" + df_data['28d抗压强度'][m] + "'").first(),
                    name="预测"
                )
                rel5 = Relationship(
                    self.matcher.match('矿渣').where("_.name=" + "'" + df_data['name'][m] + "'").first(),
                    "预测",
                    self.matcher.match('28d抗压强度').where("_.name=" + "'" + df_data['28d抗压强度'][m] + "'").first(),
                    name="预测"
                )
                rel6 = Relationship(
                    self.matcher.match('硅灰').where("_.name=" + "'" + df_data['name'][m] + "'").first(),
                    "预测",
                    self.matcher.match('28d抗压强度').where("_.name=" + "'" + df_data['28d抗压强度'][m] + "'").first(),
                    name="预测"
                )
                rel7 = Relationship(
                    self.matcher.match('水掺量').where("_.name=" + "'" + df_data['水掺量'][m] + "'").first(),
                    "相关联",
                    self.matcher.match('水灰比').where("_.name=" + "'" + df_data['水灰比'][m] + "'").first(),
                    name="相关联"
                )
                rels = Subgraph(relationships=[rel, rel1, rel2, rel3, rel4, rel5, rel6, rel7])
                self.graph.create(rels)
            except AttributeError as e:
                print(e, m)