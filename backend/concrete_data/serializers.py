from .models import ConcreteAllGene, CementGene, WaterResGene, AeaGene, AgregateCoarseGene, AgregateFineGene, AdmixtureGene, FaGene, SlagGene, SlagFineGene, SilicaFumeGene, LspGene, OxidePercentage, PoreGene, ImpermeabilityGene, CarbonizationGene, FrostResistanceWaterGene, PaperSource
from dvadmin.utils.serializers import CustomModelSerializer
# 文献信息表


class PaperSourceModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = PaperSource
        fields = "__all__"


class PaperSourceModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = PaperSource
        fields = '__all__'
# 抗冻性基因表


class FrostResistanceWaterGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = FrostResistanceWaterGene
        fields = "__all__"


class FrostResistanceWaterGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = FrostResistanceWaterGene
        fields = '__all__'
# 碳化深度基因表


class CarbonizationGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = CarbonizationGene
        fields = "__all__"


class CarbonizationGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = CarbonizationGene
        fields = '__all__'
# 氯离子渗透性基因表


class ImpermeabilityGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = ImpermeabilityGene
        fields = "__all__"


class ImpermeabilityGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = ImpermeabilityGene
        fields = '__all__'
# 孔隙率基因表


class PoreGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = PoreGene
        fields = "__all__"


class PoreGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = PoreGene
        fields = '__all__'

# 氧化物百分比表(硅灰表SilicaFumeGene，超细矿渣表SlagFineGene，矿渣表SlagGene，粉煤灰表FaGene，水泥表CementGene，石灰石表LspGene)


class OxidePercentageModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = OxidePercentage
        fields = "__all__"


class OxidePercentageModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = OxidePercentage
        fields = '__all__'

    def create(self, validated_data):
        print(validated_data)
        # oxide_data = validated_data.pop('dept_belong_id', None)
        validated_data.pop('dept_belong_id', None)
        print(validated_data)
        return OxidePercentage.objects.create(**validated_data)

# 石灰石基因表


class LspGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """
    # oxide_lsp = OxidePercentageModelSerializer()

    class Meta:
        model = LspGene
        fields = "__all__"

    # def create(self, validated_data):
    #     oxide_lsp_data = validated_data.pop('oxide_lsp')
    #     oxide_lsp = OxidePercentage.objects.create(**oxide_lsp_data)
    #     lsp_gene = LspGene.objects.create(
    #         oxide_lsp=oxide_lsp, **validated_data)
    #     return lsp_gene


class LspGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """
    # oxide_lsp = OxidePercentageModelSerializer()

    class Meta:
        model = LspGene
        fields = '__all__'

    # def create(self, validated_data):
    #     oxide_lsp_data = validated_data.pop('oxide_lsp')
    #     oxide_lsp = OxidePercentage.objects.create(**oxide_lsp_data)
    #     lsp_gene = LspGene.objects.create(
    #         oxide_lsp=oxide_lsp, **validated_data)
    #     return lsp_gene
# 硅灰基因表


class SilicaFumeGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """
    # oxide_silica = OxidePercentageModelSerializer()

    class Meta:
        model = SilicaFumeGene
        fields = "__all__"

    # def create(self, validated_data):
    #     oxide_silica_data = validated_data.pop('oxide_silica')
    #     oxide_silica = OxidePercentage.objects.create(**oxide_silica_data)
    #     silica_fume_gene = SilicaFumeGene.objects.create(
    #         oxide_silica=oxide_silica, **validated_data)
    #     return silica_fume_gene


class SilicaFumeGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """
    # oxide_silica = OxidePercentageModelCreateUpdateSerializer()

    class Meta:
        model = SilicaFumeGene
        fields = '__all__'

    # def create(self, validated_data):
    #     oxide_silica_data = validated_data.pop('oxide_silica')
    #     oxide_silica = OxidePercentage.objects.create(**oxide_silica_data)
    #     silica_fume_gene = SilicaFumeGene.objects.create(
    #         oxide_silica=oxide_silica, **validated_data)
    #     return silica_fume_gene

# 超细矿渣基因表


class SlagFineGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """
    # oxide_slag_fine = OxidePercentageModelSerializer()

    class Meta:
        model = SlagFineGene
        fields = "__all__"

    # def create(self, validated_data):
    #     oxide_slag_fine_data = validated_data.pop('oxide_slag_fine')
    #     oxide_slag_fine = OxidePercentage.objects.create(
    #         **oxide_slag_fine_data)
    #     slag_fine_gene = SlagFineGene.objects.create(
    #         oxide_slag_fine=oxide_slag_fine, **validated_data)
    #     return slag_fine_gene


class SlagFineGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """
    # oxide_slag_fine = OxidePercentageModelCreateUpdateSerializer()

    class Meta:
        model = SlagFineGene
        fields = '__all__'

    # def create(self, validated_data):
    #     oxide_slag_fine_data = validated_data.pop('oxide_slag_fine')
    #     oxide_slag_fine = OxidePercentage.objects.create(
    #         **oxide_slag_fine_data)
    #     slag_fine_gene = SlagFineGene.objects.create(
    #         oxide_slag_fine=oxide_slag_fine, **validated_data)
    #     return slag_fine_gene
# 矿渣基因表


class SlagGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """
    # oxide_slag = OxidePercentageModelSerializer()

    class Meta:
        model = SlagGene
        fields = "__all__"

    # def create(self, validated_data):
    #     oxide_slag_data = validated_data.pop('oxide_slag')
    #     oxide_slag = OxidePercentage.objects.create(**oxide_slag_data)
    #     slag_gene = SlagGene.objects.create(
    #         oxide_slag=oxide_slag, **validated_data)
    #     return slag_gene


class SlagGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """
    # oxide_slag = OxidePercentageModelCreateUpdateSerializer()

    class Meta:
        model = SlagGene
        fields = '__all__'

    # def create(self, validated_data):
    #     oxide_slag_data = validated_data.pop('oxide_slag')
    #     oxide_slag = OxidePercentage.objects.create(**oxide_slag_data)
    #     slag_gene = SlagGene.objects.create(
    #         oxide_slag=oxide_slag, **validated_data)
    #     return slag_gene
# 粉煤灰基因表


class FaGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """
    # oxide_fa = OxidePercentageModelSerializer()

    class Meta:
        model = FaGene
        fields = "__all__"

    # def create(self, validated_data):
    #     oxide_fa_data = validated_data.pop('oxide_fa')
    #     oxide_fa = OxidePercentage.objects.create(**oxide_fa_data)
    #     fa_gene = FaGene.objects.create(oxide_fa=oxide_fa, **validated_data)
    #     return fa_gene


class FaGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """
    # oxide_fa = OxidePercentageModelCreateUpdateSerializer()

    class Meta:
        model = FaGene
        fields = '__all__'

    # def create(self, validated_data):
    #     oxide_fa_data = validated_data.pop('oxide_fa')
    #     oxide_fa = OxidePercentage.objects.create(**oxide_fa_data)
    #     fa_gene = FaGene.objects.create(oxide_fa=oxide_fa, **validated_data)
    #     return fa_gene

# 掺合料基因表（粉煤灰，矿渣，超细矿渣，硅灰，石灰石）


# 粗骨料基因表

class AgregateCoarseGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = AgregateCoarseGene
        fields = "__all__"


class AgregateCoarseGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = AgregateCoarseGene
        fields = '__all__'

# 细骨料基因表


class AgregateFineGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = AgregateFineGene
        fields = "__all__"


class AgregateFineGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = AgregateFineGene
        fields = '__all__'


# 水泥表


class CementGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """
    # oxide_cement = OxidePercentageModelSerializer()

    class Meta:
        model = CementGene
        fields = "__all__"

    # def create(self, validated_data):
    #     oxide_cement_data = validated_data.pop('oxide_cement')
    #     oxide_cement = OxidePercentage.objects.create(**oxide_cement_data)
    #     cement_gene = CementGene.objects.create(
    #         oxide_cement=oxide_cement, **validated_data)
    #     return cement_gene


class CementGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """
    # oxide_cement = OxidePercentageModelSerializer()

    class Meta:
        model = CementGene
        fields = '__all__'

    # def create(self, validated_data):
    #     oxide_cement_data = validated_data.pop('oxide_cement')
    #     oxide_cement = OxidePercentage.objects.create(**oxide_cement_data)
    #     cement_gene = CementGene.objects.create(
    #         oxide_cement=oxide_cement, **validated_data)
    #     return cement_gene

# 减水剂


class WaterResGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = WaterResGene
        fields = "__all__"


class WaterResGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = WaterResGene
        fields = '__all__'

# AeaGene引气剂


class AeaGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = AeaGene
        fields = "__all__"


class AeaGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = AeaGene
        fields = '__all__'


class AdmixtureGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """
    # fa = FaGeneModelSerializer()
    # slag = SlagGeneModelSerializer()
    # slag_fine = SlagFineGeneModelSerializer()
    # silica_fume = SilicaFumeGeneModelSerializer()
    # lsp = LspGeneModelSerializer()

    class Meta:
        model = AdmixtureGene
        fields = "__all__"

    # def create(self, validated_data):
    #     fa_data = validated_data.pop('fa')
    #     slag_data = validated_data.pop('slag')
    #     slag_fine_data = validated_data.pop('slag_fine')
    #     silica_fume_data = validated_data.pop('silica_fume')
    #     lsp_data = validated_data.pop('lsp')

    #     fa = FaGene.objects.create(**fa_data)
    #     slag = SlagGene.objects.create(**slag_data)
    #     slag_fine = SlagFineGene.objects.create(**slag_fine_data)
    #     silica_fume = SilicaFumeGene.objects.create(**silica_fume_data)
    #     lsp = LspGene.objects.create(**lsp_data)

    #     admixture_gene = AdmixtureGene.objects.create(
    #         fa=fa,
    #         slag=slag,
    #         slag_fine=slag_fine,
    #         silica_fume=silica_fume,
    #         lsp=lsp,
    #         **validated_data
    #     )
    #     return admixture_gene


class AdmixtureGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """
    # fa = FaGeneModelCreateUpdateSerializer()
    # slag = SlagGeneModelCreateUpdateSerializer()
    # slag_fine = SlagFineGeneModelCreateUpdateSerializer()
    # silica_fume = SilicaFumeGeneModelCreateUpdateSerializer()
    # lsp = LspGeneModelCreateUpdateSerializer()

    class Meta:
        model = AdmixtureGene
        fields = '__all__'

    # def create(self, validated_data):
    #     fa_data = validated_data.pop('fa')
    #     slag_data = validated_data.pop('slag')
    #     slag_fine_data = validated_data.pop('slag_fine')
    #     silica_fume_data = validated_data.pop('silica_fume')
    #     lsp_data = validated_data.pop('lsp')

    #     fa = FaGene.objects.create(**fa_data)
    #     slag = SlagGene.objects.create(**slag_data)
    #     slag_fine = SlagFineGene.objects.create(**slag_fine_data)
    #     silica_fume = SilicaFumeGene.objects.create(**silica_fume_data)
    #     lsp = LspGene.objects.create(**lsp_data)

    #     admixture_gene = AdmixtureGene.objects.create(
    #         fa=fa,
    #         slag=slag,
    #         slag_fine=slag_fine,
    #         silica_fume=silica_fume,
    #         lsp=lsp,
    #         **validated_data
    #     )
    #     return admixture_gene

# 全息基因表


class ConcreteAllGeneModelSerializer(CustomModelSerializer):
    """
    序列化器
    """
    # cement = CementGeneModelSerializer()
    # water_res = WaterResGeneModelSerializer()
    # aea = AeaGeneModelSerializer()
    # agregate_coarse = AgregateCoarseGeneModelSerializer()
    # agregate_fine = AgregateFineGeneModelSerializer()
    # admixture = AdmixtureGeneModelSerializer()
    # pore_characteristics = PoreGeneModelSerializer()
    # carbonization = CarbonizationGeneModelSerializer()
    # frost_resistancer = FrostResistanceWaterGeneModelSerializer()
    # impermeability = ImpermeabilityGeneModelSerializer()
    # paper = PaperSourceModelSerializer()
    # oxide_percentage = OxidePercentageModelSerializer()

    class Meta:
        model = ConcreteAllGene
        fields = "__all__"

    # def create(self, validated_data):
    #     cement_data = validated_data.pop('cement')
    #     water_res_data = validated_data.pop('water_res')
    #     aea_data = validated_data.pop('aea')
    #     agregate_coarse_data = validated_data.pop('agregate_coarse')
    #     agregate_fine_data = validated_data.pop('agregate_fine')
    #     admixture_data = validated_data.pop('admixture')
    #     pore_characteristics_data = validated_data.pop('pore_characteristics')
    #     carbonization_data = validated_data.pop('carbonization')
    #     frost_resistancer_data = validated_data.pop('frost_resistancer')
    #     impermeability_data = validated_data.pop('impermeability')
    #     paper_data = validated_data.pop('paper')

    #     cement = CementGene.objects.create(**cement_data)
    #     water_res = WaterResGene.objects.create(**water_res_data)
    #     aea = AeaGene.objects.create(**aea_data)
    #     agregate_coarse = AgregateCoarseGene.objects.create(
    #         **agregate_coarse_data)
    #     agregate_fine = AgregateFineGene.objects.create(**agregate_fine_data)
    #     admixture = AdmixtureGene.objects.create(**admixture_data)
    #     pore_characteristics = PoreGene.objects.create(
    #         **pore_characteristics_data)
    #     carbonization = CarbonizationGene.objects.create(**carbonization_data)
    #     frost_resistancer = FrostResistanceWaterGene.objects.create(
    #         **frost_resistancer_data)
    #     impermeability = ImpermeabilityGene.objects.create(
    #         **impermeability_data)
    #     paper = PaperSource.objects.create(**paper_data)

    #     concrete_all_gene = ConcreteAllGene.objects.create(
    #         cement=cement,
    #         water_res=water_res,
    #         aea=aea,
    #         agregate_coarse=agregate_coarse,
    #         agregate_fine=agregate_fine,
    #         admixture=admixture,
    #         pore_characteristics=pore_characteristics,
    #         carbonization=carbonization,
    #         frost_resistancer=frost_resistancer,
    #         impermeability=impermeability,
    #         paper=paper,
    #         **validated_data
    #     )
    #     return concrete_all_gene


class ConcreteAllGeneModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """
    # cement = CementGeneModelSerializer()
    # water_res = WaterResGeneModelSerializer()
    # aea = AeaGeneModelSerializer()
    # agregate_coarse = AgregateCoarseGeneModelSerializer()
    # agregate_fine = AgregateFineGeneModelSerializer()
    # admixture = AdmixtureGeneModelSerializer()
    # pore_characteristics = PoreGeneModelSerializer()
    # carbonization = CarbonizationGeneModelSerializer()
    # frost_resistancer = FrostResistanceWaterGeneModelSerializer()
    # impermeability = ImpermeabilityGeneModelSerializer()
    # paper = PaperSourceModelSerializer()

    class Meta:
        model = ConcreteAllGene
        fields = '__all__'

    def create(self, validated_data):
        print(validated_data)
        validated_data.pop('dept_belong_id', None)
        return ConcreteAllGene.objects.create(**validated_data)
    #     # print(validated_data)
    #     cement_data = validated_data.pop('cement')
    #     cement_data.pop('dept_belong_id', None)
    #     # dept_belong_id = validated_data.pop('dept_belong_id', None)
    #     # print(cement_data)
    #     water_res_data = validated_data.pop('water_res')
    #     water_res_data.pop('dept_belong_id', None)
    #     aea_data = validated_data.pop('aea')
    #     aea_data.pop('dept_belong_id', None)
    #     agregate_coarse_data = validated_data.pop('agregate_coarse')
    #     agregate_coarse_data.pop('dept_belong_id', None)
    #     agregate_fine_data = validated_data.pop('agregate_fine')
    #     agregate_fine_data.pop('dept_belong_id', None)
    #     admixture_data = validated_data.pop('admixture')
    #     admixture_data.pop('dept_belong_id', None)
    #     pore_characteristics_data = validated_data.pop('pore_characteristics')
    #     pore_characteristics_data.pop('dept_belong_id', None)
    #     carbonization_data = validated_data.pop('carbonization')
    #     carbonization_data.pop('dept_belong_id', None)
    #     frost_resistancer_data = validated_data.pop('frost_resistancer')
    #     frost_resistancer_data.pop('dept_belong_id', None)
    #     impermeability_data = validated_data.pop('impermeability')
    #     impermeability_data.pop('dept_belong_id', None)
    #     paper_data = validated_data.pop('paper')
    #     paper_data.pop('dept_belong_id', None)

    #     oxide_cement_data = cement_data.pop('oxide_cement')
    #     # 从 oxide_cement_data 中删除 dept_belong_id
    #     oxide_cement_data.pop('dept_belong_id', None)
    #     oxide_cement = OxidePercentage.objects.create(**oxide_cement_data)
    #     # print(oxide_cement_data)
    #     cement = CementGene.objects.create(
    #         oxide_cement=oxide_cement, **cement_data)
    #     water_res = WaterResGene.objects.create(**water_res_data)
    #     aea = AeaGene.objects.create(**aea_data)
    #     agregate_coarse = AgregateCoarseGene.objects.create(
    #         **agregate_coarse_data)
    #     agregate_fine = AgregateFineGene.objects.create(**agregate_fine_data)
    #     admixture = AdmixtureGene.objects.create(**admixture_data)
    #     pore_characteristics = PoreGene.objects.create(
    #         **pore_characteristics_data)
    #     carbonization = CarbonizationGene.objects.create(**carbonization_data)
    #     frost_resistancer = FrostResistanceWaterGene.objects.create(
    #         **frost_resistancer_data)
    #     impermeability = ImpermeabilityGene.objects.create(
    #         **impermeability_data)
    #     paper = PaperSource.objects.create(**paper_data)

    #     concrete_all_gene = ConcreteAllGene.objects.create(
    #         cement=cement,
    #         water_res=water_res,
    #         aea=aea,
    #         agregate_coarse=agregate_coarse,
    #         agregate_fine=agregate_fine,
    #         admixture=admixture,
    #         pore_characteristics=pore_characteristics,
    #         carbonization=carbonization,
    #         frost_resistancer=frost_resistancer,
    #         impermeability=impermeability,
    #         paper=paper,
    #         **validated_data
    #     )
    #     return concrete_all_gene
