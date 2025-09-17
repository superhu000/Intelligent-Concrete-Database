url = 'bolt://localhost:7687'
username = "neo4j"
password = "123456789"
# database_name = "neo4j"


# 属性表
# 单属性标签
property_Alone = [
    ('水掺量', 7),
    ('水灰比', 8),
    ('孔隙率', 40),
    ('28d抗压强度', 44),
]

# 水泥标签
property_Cement = [
    ('name', 0),
    ('C_CaO', 1),
    ('C_SiO2', 2),
    ('C_Al2O3', 3),
    ('C_MgO', 4),
    ('C_Fe2O3', 5),
    ('水泥掺量', 6),
]

# 粉煤灰标签（Fly ash）
property_FlyAsh = [
    ('name', 0),
    ('FlyAsh_CaO', 20),
    ('FlyAsh_SiO2', 21),
    ('FlyAsh_Al2O3', 22),
    ('FlyAsh_MgO', 23),
    ('FlyAsh_Fe2O3', 24),
    ('FlyAsh_I级掺量', 25),
    ('FlyAsh_II级掺量', 26),
    ('FlyAsh掺量', 27),
]

# 矿渣标签（slag）
property_Slag = [
    ('name', 0),
    ('Slag_CaO', 28),
    ('Slag_SiO2', 29),
    ('Slag_Al2O3', 30),
    ('Slag_MgO', 31),
    ('Slag_Fe2O3', 32),
    ('Slag掺量', 33),
]

# 硅灰标签（SilicaFume->SF）
property_SF = [
    ('name', 0),
    ('SF_CaO', 34),
    ('SF_SiO2', 35),
    ('SF_Al2O3', 36),
    ('SF_MgO', 37),
    ('SF_Fe2O3', 38),
    ('SF掺量', 39),
]

# 所有属性
property_All = [  # 现在并不完整
    ('name', 0), ('C_CaO', 1), ('C_SiO2', 2), ('C_Al2O3', 3), ('C_MgO', 4), ('C_Fe2O3', 5), ('水泥掺量', 6),
    ('水掺量', 7), ('水灰比', 8),
    ('FlyAsh_CaO', 20), ('FlyAsh_SiO2', 21), ('FlyAsh_Al2O3', 22), ('FlyAsh_MgO', 23), ('FlyAsh_Fe2O3', 24),
    ('FlyAsh_I级掺量', 25), ('FlyAsh_II级掺量', 26), ('FlyAsh掺量', 27),
    ('Slag_CaO', 28), ('Slag_SiO2', 29), ('Slag_Al2O3', 30), ('Slag_MgO', 31), ('Slag_Fe2O3', 32), ('Slag掺量', 33),
    ('SF_CaO', 34), ('SF_SiO2', 35), ('SF_Al2O3', 36), ('SF_MgO', 37), ('SF_Fe2O3', 38), ('SF掺量', 39),
    ('孔隙率', 40), ('28d抗压强度', 44),
]

# id C-CaO	C-SiO2	C-Al2O3	C-MgO	C-Fe2O3	水泥掺量	水掺量	水灰比	减水剂掺量	碎石1	碎石2	碎石3	碎石4	碎石5	碎石6	碎石7	碎石掺量	河砂细度模数	河砂掺量	MH-CaO	MH-SiO2	MH-Al2O3	MH-MgO	MH-Fe2O3	MH-I级掺量	MH-II级掺量	MH掺量	KZ-CaO	KZ-SiO2	KZ-Al2O3	KZ-MgO	KZ-Fe2O3	KZ-掺量	GF-CaO	GF-SiO2	GF-Al2O3	GF-MgO	GF-Fe2O3	GF-掺量	孔隙率	扩散系数	渗透系数	电通量	28d抗压强度	碎石总用量	MH总用量
