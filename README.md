# Projeto de Detecção de Objetos em Restaurante com Machine Learning

## Descrição do Projeto
Este projeto utiliza redes neurais, especificamente a arquitetura YOLOv8, para detectar objetos em um ambiente de restaurante. O objetivo é identificar elementos como pessoas, garçons, mesas e pratos, aplicando técnicas avançadas de visão computacional e aprendizado de máquina. O modelo é treinado para realizar detecções em tempo real, oferecendo potenciais aplicações práticas na otimização do serviço e gerenciamento de espaço em restaurantes.

## Configuração do Ambiente
Para configurar o ambiente de desenvolvimento necessário para este projeto, siga os passos abaixo:

1. Instale os pacotes Python necessários:
   ```bash
   pip install matplotlib pycocotools opencv-python-headless keras-cv keras-core

2. Clone este repositório para obter os scripts e os notebooks Jupyter:
   ```bash
   git clone https://github.com/vitorbandeira1/ml_projeto.git

## Estrutura do Projeto
- rest_test.ipynb: Jupyter notebook contendo o código principal do projeto, incluindo a preparação dos dados, arquitetura do modelo, treinamento e inferência.
- infer_utils.py: Script Python para executar o modelo treinado em imagens estáticas.
- infer_video.py: Script Python para executar o modelo treinado em vídeos.

## Executando o Projeto
Para executar o projeto e ver a detecção de objetos em ação, siga estes passos:

1. Abra o Jupyter Notebook (rest_test.ipynb) e execute todas as células para treinar o modelo ou carregue um modelo pré-treinado.

2. Para testar o modelo em imagens estáticas, use o script infer_utils.py:
   ```bash
    python infer_utils.py [caminho_para_imagem]
    
3. Para testar o modelo em vídeos, use o script infer_video.py:
   ```bash
    python infer_video.py [caminho_para_vídeo]
    
Você pode baixar um vídeo de teste de restaurante a partir do seguinte link: [https://drive.google.com/file/d/1YkEiQyWL1E3QSmfdNmESihPA0JAwWDYV/view]Vídeo de Restaurante para Teste.