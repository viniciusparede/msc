from mmsdk import mmdatasdk as md

# caminho para o arquivo .csd
csd_path = "/home/vinicius/Documentos/Repositories/msc/data"

# carregar o arquivo .csd como um novo conjunto de dados
dataset = md.mmdataset(csd_path)

dataset.align("Opinion Segment Labels")

# agora você pode trabalhar com 'dataset' conforme necessário
