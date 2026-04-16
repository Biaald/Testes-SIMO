# Simulação de Desempenho LoRa com Diversidade Espacial

Este projeto faz parte de uma pesquisa de iniciação científica (PIBIC) focada na análise da Taxa de Erro de Bit (BER) do protocolo LoRa em canais com desvanecimento, utilizando técnicas de diversidade no receptor (SIMO - Single Input Multiple Output).

## Visão Geral do Código

O simulador foi desenvolvido em Python, utilizando processamento vetorizado com a biblioteca `NumPy` para garantir alta performance. A ferramenta permite comparar diferentes técnicas de combinação (MRC, EGC, SC), fatores de espalhamento (SF) e condições de canal realistas.

## Conceitos Principais

**1. Mini-Batches (Processamento em Lotes)**

Devido à natureza da modulação LoRa, simular um grande número de pontos (200k símbolos) exige uma quantidade massiva de memória RAM.

- O Desafio: Para cada símbolo, o receptor processa uma FFT de tamanho $2^{SF}$ para cada uma das $L$ antenas. Simular 200k pontos de uma vez exigiria $>12$ GB de RAM, excedendo a capacidade de processamento.
- A Solução: Implementamos a lógica de Mini-Batches. O código divide os 200k símbolos em 10 ou 20 blocos menores. Cada bloco é processado, os erros são contabilizados, e a memória é liberada antes do próximo lote. Isso permite simular milhões de pontos mantendo o uso de RAM.

**2. Shadowing (Sombreamento Lognormal)**

Enquanto o desvanecimento Rayleigh modela pequenas variações rápidas do sinal (multipropagação), o Shadowing modela a perda de potência em larga escala causada por obstáculos físicos (prédios, árvores).

- Modelo: Utiliza uma distribuição Lognormal com desvio padrão $\sigma$.I
- mpacto: Introduz uma variação na potência média recebida. No código, isso é simulado multiplicando o ganho do canal por um fator aleatório $10^{X/20}$, onde $X \sim \mathcal{N}(0, \sigma^2)$.

**3. Desequilíbrio de Ganho entre Antenas**

Em cenários teóricos, assume-se que todas as antenas do receptor recebem o sinal com a mesma potência média. Na prática, uma antena pode estar melhor posicionada que outra.

- Simulação: O código aplica um vetor de ganho escalar às ramificações do receptor.
- Objetivo: Avaliar a robustez do algoritmo MRC (que deve dar mais peso à antena mais forte) versus o SC (que apenas seleciona a melhor) em condições de instalação imperfeita.

## Implementação Técnica

Para viabilizar as simulações, foram aplicadas as seguintes otimizações:

**Vetorização Total**: Evitamos loops for do Python para processar símbolos. Operações como a geração de chirps e o cálculo do canal são feitas diretamente em matrizes multidimensionais do `NumPy`.

`symbols_to_bits`: Uma função otimizada que converte os índices dos símbolos LoRa em bits utilizando operações bitwise, essencial para o cálculo rápido do BER.

Gerenciamento de Memória: Uso do comando `del` e `gc.collect()` para garantir que matrizes pesadas de ruído AWGN sejam descartadas logo após o uso.

Estimação de Canal: Implementação do MRC (Maximal Ratio Combining) utilizando o produto escalar entre o conjugado do canal estimado e o sinal recebido, maximizando a relação sinal-ruído (SNR) resultante.

## Modelagem Matemática: Canal de Desvanecimento Composto e Combinação MRC

Para garantir a validade dos resultados em cenários reais de redes LPWAN (Low Power Wide Area Network), a simulação modela o ambiente de propagação como um Canal de Desvanecimento Composto (Composite Fading Channel). Este modelo une os efeitos de pequena e larga escala simultaneamente.

**1. O Canal Composto**

O sinal transmitido sofre dois fenômenos estocásticos independentes antes de chegar às $L$ antenas do receptor.

O ganho do canal para a $i$-ésima antena é dado por:

$$h_{total, i} = h_{shadow} \cdot h_{rayleigh, i}$$

**Onde:**

- $h_{rayleigh, i}$ (Desvanecimento de Pequena Escala): Representa a multipropagação (reflexões e espalhamentos rápidos). É modelado como uma variável aleatória complexa Gaussiana com média zero e variância unitária: $h_{rayleigh, i} \sim \mathcal{CN}(0, 1)$. Assumimos desvanecimento plano e independente para cada antena.
- $h_{shadow}$ (Sombreamento Lognormal): Representa o desvanecimento de larga escala causado pelo bloqueio físico de obstáculos (prédios, árvores). Como as antenas do receptor estão fisicamente próximas, o obstáculo atenua todo o arranjo igualmente. Modelamos uma variável Gaussiana real $X \sim \mathcal{N}(0, \sigma^2)$, onde $\sigma$ é o desvio padrão em dB, e a convertemos para a escala linear de amplitude:

$$h_{shadow} = 10^{\frac{X}{20}}$$

**2. O Sinal Recebido**

Seja $s$ o símbolo LoRa transmitido com energia $E_s$. O sinal na banda base recebido $r_i$ na antena $i$ é afetado pelo canal composto e pelo ruído térmico aditivo:

$$r_i = h_{total, i} \sqrt{E_s} s + n_i$$

Onde $n_i \sim \mathcal{CN}(0, N_0)$ é o Ruído Branco Gaussiano Aditivo (AWGN) presente no circuito do receptor.

3. A Técnica MRC (Maximal Ratio Combining)

O MRC é o combinador linear ótimo. Ele requer conhecimento perfeito do estado do canal (CSI - Channel State Information). O receptor multiplica o sinal de cada ramificação pelo conjugado complexo do canal estimado ($h_{total, i}^*$) e soma os resultados, maximizando a Relação Sinal-Ruído (SNR) na saída:

$$y = \frac{\sum_{i=1}^{L} h_{total, i}^* r_i}{\sum_{i=1}^{L} |h_{total, i}|^2}$$

Esta operação cancela perfeitamente a distorção de fase ($h^* \cdot h = |h|^2$) e soma as amplitudes de forma construtiva, dando maior peso estatístico às antenas que possuem instantaneamente a melhor qualidade de sinal.

## Implementação Computacional (Algoritmo Vetorizado)

Para viabilizar a simulação de centenas de milhares de símbolos sem exceder a memória RAM, as equações matemáticas foram mapeadas diretamente para operações matriciais vetorizadas na biblioteca NumPy utilizando processamento em mini-batches.

**Passo A**: Geração do Sombreamento ($h_{shadow}$) 
Sorteamos os valores da distribuição Normal e convertemos para ganho linear. O bloqueio afeta todo o arranjo, logo, gera-se um valor único por símbolo:
 ```
Python
s_db = np.random.normal(0, sigma_shadow, size=(batch_size, 1))
shadow_gain = 10 ** (s_db / 20)
```

**Passo B**: Geração do Desvanecimento Rápido ($h_{rayleigh}$) 

Criamos a matriz de números complexos aleatórios independentes, alocando uma coluna para cada antena ($num\_rx$):

```
Python
hs = (1/np.sqrt(2)) * (np.random.randn(batch_size, num_rx) + 1j * np.random.randn(batch_size, num_rx))
```

**Passo C**: O Canal Composto ($h_{total}$) 
Através do broadcasting do NumPy, multiplicamos a matriz de multipropagação pelo ganho de sombreamento simultaneamente para todas as antenas:

```
Python
h_total = hs * shadow_gain
```

**Passo D**: Aplicação do MRC O combinador é calculado somando o produto conjugado ao longo do eixo das antenas (axis=1). O numerador executa o alinhamento de fase e o denominador realiza a normalização da potência:

```
Python
num = np.sum(np.conjugate(h_total)[:, :, None] * rs, axis=1)
den = np.sum(np.abs(h_total)**2, axis=1, keepdims=True)
combined = num / den
```
