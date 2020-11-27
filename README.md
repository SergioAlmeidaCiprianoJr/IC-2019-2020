# Introdução

Neste projeto, trabalhamos com o método de gradientes conjugados (MGC) para solução
de um subproblema do método de Newton para minimização irrestrita algoritmo que
chamamos de Newton-MGC. Nosso objetivo é conceber um estudo introdutório à área de
métodos computacionais de otimização com o algoritmo de gradientes conjugados, um
método queintegra a classe de algoritmos clássicos para otimização e permanece um
assunto de pesquisa relevante até os dias de hoje. O presente trabalho parte da
formulação do problema de minimização sem restrições, uma introdução aos métodos de
direções conjugadas, alguns conceitos pré-requisitos para a plena compreensão dos
gradientes conjugados e sua integração com o método de Newton para a solução de
problemas de minimização. Ademais, foi feita uma implementação do método estudado
na linguagem Julia e experimentos numéricos permitiram a validação do algoritmo
estudado, além de comparações com outro método da literatura.

# Metodologia

A metodologia consistiu: (i) no estudo e compreensão dos métodos trabalhados com a
leitura de livros e artigos apresentados na bibliografia, paralelamente a conceitos
básicos de otimização não linear; e novas tecnologias para aplicar computacionalmente
o que foi estudado; (ii) implementação do método proposto com as tecnologias estudadas;
(iii) experimentações numéricas que validaram os métodos implementados e (iv) análise
e interpretação dos resultados experimentais por meio de tabelas e de perfis de
desempenho.

# Resultados

Experimentos numéricos com problemas clássicos da literatura demonstraram a robustez
do MGC. Podemos fazer as seguintes observações perante os resultados: em relação ao
consumo de memória, Newton-MGC mostrou-se eficaz, o que era esperado dado que uma das
propriedades mais interessantes do MGC é de não precisar armazenar todasas direções
anteriores para garantir que novas direções sejam A-ortogonais; quanto aquantidade de
iterações, no algoritmo de Newton-Cholesky as iterações do subproblema escalaram
rapidamente a medida que o número de variáveis aumentava. Diante disso, todos os casos
com 10000 variáveis excederam o limite de 10 minutos na primeira iteração. Em
contrapartida, o desempenho do Newton-MGC continuou sendo muito bom e seu pior
desempenho foi de 4 segundos em um caso com 10000 variáveis,mostrando assim sua
robustez para problemas de grande porte.

# Conclusão

Com os experimentos numéricos, podemos dizer que aplicar o método de gradientes
conjugados ao método de Newton mostrou ser uma estratégia adequada para a solução de
problemas de minimização irrestrita de grande porte. Em especial, apresentou bons
resultados de desempenho nos problemas com grande quantidade de variáveis. Diante
disso, vimos que o método de gradientes conjugados mostrou robustez nos quesitos
analisados e, à vista de suas propriedades, mostrou-se um campo fértil para avanços.
