# Arvore-de-decisão-multithread
Trabalho acadêmico da materia de Computação Paralela
-----

# Avaliação de Desempenho de Árvore de Decisão Paralela (Sequencial, OpenMP, MPI+OpenMP)

Este repositório contém implementações de um algoritmo de árvore de decisão C4.5 (baseado em Gain Ratio) com diferentes estratégias de paralelização para fins de avaliação de desempenho.

## 1\. Objetivo do Projeto

O objetivo deste trabalho é implementar e avaliar a performance de diferentes estratégias de paralelização (OpenMP e MPI Híbrido) para o treinamento de uma árvore de decisão, comparando-as com uma implementação sequencial.

O problema de classificação específico é **prever se um voo de uma companhia aérea atrasará por mais de 6 horas**, com base em dados de voos.

## 2\. Dataset

O conjunto de dados utilizado é o **"On-Time Delay Causes"** do Bureau of Transportation Statistics (BTS). Os arquivos `dt_train.txt` e `dt_test_with_labels.txt` são versões pré-processadas deste conjunto de dados.

  * **Fonte Original:** [https://www.transtats.bts.gov/OT\_Delay/OT\_DelayCause1.asp?20=E](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp?20=E)

## 3\. Implementações no Repositório

Este repositório inclui quatro versões do código:

1.  `trabalhoSequencial.cpp`: Versão base, totalmente sequencial.
2.  `trabalhoOpenMp.cpp`: Versão paralela com OpenMP (Nível 1 - Grão Grosso), que paraleliza o loop de seleção de atributos (`getSelectedAttribute`).
3.  `trabalhoOpenMpAninhado.cpp`: Versão paralela com OpenMP (Nível 1 + Nível 2 - Grão Fino Aninhado), que paraleliza tanto a seleção de atributos quanto os cálculos internos de entropia (`getInfoD`, `getInfoAttrD`, etc.).
4.  `trabalhoOpenMpMPI.cpp`: Versão híbrida (MPI + OpenMP), que divide o trabalho de seleção de atributos entre processos MPI (Grão Grosso) e usa OpenMP para os cálculos de entropia (Grão Fino) dentro de cada processo.

## 4\. Setup e Compilação (macOS)

As instruções abaixo assumem um ambiente macOS com **Homebrew** instalado.

### Pré-requisitos

Você precisará do `clang++` (Xcode Command Line Tools), `libomp` (para OpenMP) e `open-mpi`:

```bash
brew install libomp open-mpi
```

### Comandos de Compilação

Para cada versão, use o seguinte comando no seu terminal:

**1. Versão Sequencial (`trabalhoSequencial.cpp`)**

```bash
clang++ -std=c++11 -o exec_seq trabalhoSequencial.cpp -lm
```

**2. Versão OpenMP (`trabalhoOpenMp.cpp`)**

```bash
clang++ -std=c++11 -Xpreprocessor -fopenmp trabalhoOpenMp.cpp \
-o exec_omp \
-I/opt/homebrew/opt/libomp/include \
-L/opt/homebrew/opt/libomp/lib -lomp -lm
```

**3. Versão OpenMP Aninhado (`trabalhoOpenMpAninhado.cpp`)**

```bash
clang++ -std=c++11 -Xpreprocessor -fopenmp trabalhoOpenMpAninhado.cpp \
-o exec_omp_aninhado \
-I/opt/homebrew/opt/libomp/include \
-L/opt/homebrew/opt/libomp/lib -lomp -lm
```

**4. Versão Híbrida MPI + OpenMP (`trabalhoOpenMpMPI.cpp`)**

```bash
mpic++ -std=c++11 -Xpreprocessor -fopenmp trabalhoOpenMpMPI.cpp \
-o exec_hibrido \
-I/opt/homebrew/opt/libomp/include \
-L/opt/homebrew/opt/libomp/lib -lomp -lm
```

## 5\. Como Executar

Use os seguintes comandos para executar cada versão. O código foi projetado para receber 3 argumentos: arquivo de treino, arquivo de teste e nome do arquivo de resultado.

**1. Executar Versão Sequencial**

```bash
./exec_seq dt_train.txt dt_test_with_labels.txt dt_result_seq.txt
```

**2. Executar Versão OpenMP (Nível 1)**

Defina o número de threads que o OpenMP deve usar dentro da main.

```bash
./exec_omp dt_train.txt dt_test_with_labels.txt dt_result_omp.txt
```

**3. Executar Versão OpenMP Aninhado (Nível 1 + 2)**

```bash
./exec_omp_aninhado dt_train.txt dt_test_with_labels.txt dt_result_omp_nested.txt
```

**4. Executar Versão Híbrida (MPI + OpenMP)**

Aqui, você define o número de **processos MPI** (com `mpirun -np`) e o número de **threads OpenMP por processo** (com `OMP_NUM_THREADS`).

```bash
# Ex: 4 processos MPI, cada um com 8 threads OpenMP (Total: 32)
export OMP_NUM_THREADS=8
mpirun -np 4 ./exec_hibrido dt_train.txt dt_test_with_labels.txt dt_result_hibrido.txt
```

## 6\. Créditos

A implementação sequencial (`trabalhoSequencial.cpp`) é uma adaptação do código C4.5 de [https://github.com/bowbowbow/DecisionTree](https://github.com/bowbowbow/DecisionTree).
