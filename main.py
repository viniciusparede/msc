from mmsdk import mmdatasdk as md

# caminho para o arquivo .csd
csd_path = "/home/vinicius/Documentos/Repositories/test-cmumosei/data/CMU_MOSEI_TimestampedWordVectors.csd"

# carregar o arquivo .csd como um novo conjunto de dados
dataset = md.mmdataset(csd_path)

# agora você pode trabalhar com 'dataset' conforme necessário
