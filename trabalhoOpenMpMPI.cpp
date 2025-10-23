#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>
#include <chrono> 
#include <omp.h>    
#include <limits>  
#include <algorithm> 
#include <cctype>  
#include <mpi.h> 

using namespace std;

class Table {
    public:
        vector<string> attrName;
        vector<vector<string>> data;

        vector<vector<string>> attrValueList;
        void extractAttrValue() {
            attrValueList.resize(attrName.size());
            for (int j = 0; j < attrName.size(); j++) {
                map<string, int> value;
                for (int i = 0; i < data.size(); i++) {
                    value[data[i][j]] = 1;
                }

                for (auto iter = value.begin(); iter != value.end(); iter++) {
                    attrValueList[j].push_back(iter->first);
                }
            }
        }
};

class Node {
    public:
        int criteriaAttrIndex;
        string attrValue;

        int treeIndex;
        bool isLeaf;
        string label;

        vector<int> children;

        Node() {
            isLeaf = false;
        }
};

class DecisionTree {
    public:
        Table initialTable;
        vector<Node> tree;

        DecisionTree(Table table) {
            initialTable = table;
            initialTable.extractAttrValue();

            Node root;
            root.treeIndex = 0;
            tree.push_back(root);
            run(initialTable, 0);

            // Apenas o Rank 0 imprime a árvore
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            if (world_rank == 0) {
                printTree(0, "");
                cout << "<-- finish generating decision tree -->" << endl << endl;
            }
        }

        string guess(vector<string> row) {
            string label = "";
            int leafNode = dfs(row, 0);
            if (leafNode == -1) {
                return "dfs failed";
            }
            label = tree[leafNode].label;
            return label;
        }

        int dfs(vector<string>& row, int here) {
            if (tree[here].isLeaf) {
                return here;
            }

            int criteriaAttrIndex = tree[here].criteriaAttrIndex;

            for (int i = 0; i < tree[here].children.size(); i++) {
                int next = tree[here].children[i];

                if (row[criteriaAttrIndex] == tree[next].attrValue) {
                    return dfs(row, next);
                }
            }
            return -1;
        }

        void run(Table table, int nodeIndex) {
            if (isLeafNode(table) == true) {
                tree[nodeIndex].isLeaf = true;
                tree[nodeIndex].label = table.data.back().back();
                return;
            }

            int selectedAttrIndex = getSelectedAttribute(table);

            if (selectedAttrIndex == -1) {
                tree[nodeIndex].isLeaf = true;
                tree[nodeIndex].label = getMajorityLabel(table).first;
                return;
            }

            map<string, vector<int>> attrValueMap;
            for (int i = 0; i < table.data.size(); i++) {
                attrValueMap[table.data[i][selectedAttrIndex]].push_back(i);
            }

            tree[nodeIndex].criteriaAttrIndex = selectedAttrIndex;

            pair<string, int> majority = getMajorityLabel(table);
            if ((double)majority.second / table.data.size() > 0.8) {
                tree[nodeIndex].isLeaf = true;
                tree[nodeIndex].label = majority.first;
                return;
            }

            for (int i = 0; i < initialTable.attrValueList[selectedAttrIndex].size(); i++) {
                string attrValue = initialTable.attrValueList[selectedAttrIndex][i];

                Table nextTable;
                vector<int> candi = attrValueMap[attrValue];
                for (int i = 0; i < candi.size(); i++) {
                    nextTable.data.push_back(table.data[candi[i]]);
                }

                Node nextNode;
                nextNode.attrValue = attrValue;
                nextNode.treeIndex = (int)tree.size();
                tree[nodeIndex].children.push_back(nextNode.treeIndex);
                tree.push_back(nextNode);

                if (nextTable.data.size() == 0) {
                    nextNode.isLeaf = true;
                    nextNode.label = getMajorityLabel(table).first;
                    tree[nextNode.treeIndex] = nextNode;
                } else {
                    run(nextTable, nextNode.treeIndex);
                }
            }
        }

        double getEstimatedError(double f, int N) {
            double z = 0.69;
            if (N == 0) {
                cout << ":: getEstimatedError :: N is zero" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            return (f + z * z / (2 * N) + z * sqrt(f / N - f * f / N + z * z / (4 * N * N))) / (1 + z * z / N);
        }

        pair<string, int> getMajorityLabel(Table table) {
            string majorLabel = "";
            int majorCount = 0;

            map<string, int> labelCount;
            for (int i = 0; i < table.data.size(); i++) {
                labelCount[table.data[i].back()]++;

                if (labelCount[table.data[i].back()] > majorCount) {
                    majorCount = labelCount[table.data[i].back()];
                    majorLabel = table.data[i].back();
                }
            }

            return {majorLabel, majorCount};
        }

        bool isLeafNode(Table table) {
            for (int i = 1; i < table.data.size(); i++) {
                if (table.data[0].back() != table.data[i].back()) {
                    return false;
                }
            }
            return true;
        }

        int getSelectedAttribute(Table table) {
            
            int world_rank, world_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);

            int numAttributes = initialTable.attrName.size() - 1; 
            
            vector<double> localGainRatios(numAttributes, 0.0);
            vector<double> globalGainRatios(numAttributes, 0.0);

            for (int i = world_rank; i < numAttributes; i += world_size) {
                localGainRatios[i] = getGainRatio(table, i);
            }

            MPI_Allreduce(localGainRatios.data(), globalGainRatios.data(),
                          numAttributes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            int maxAttrIndex = -1;
            double maxAttrValue = 0.0; 

            for (int i = 0; i < numAttributes; i++) {
                if (maxAttrValue < globalGainRatios[i]) {
                    maxAttrValue = globalGainRatios[i];
                    maxAttrIndex = i;
                }
            }

            // Todos os processos retornam o mesmo índice
            return maxAttrIndex;
        }


        double getGainRatio(Table table, int attrIndex) {
            double splitInfo = getSplitInfoAttrD(table, attrIndex);
            if (splitInfo == 0.0) {
                return 0.0;
            }
            
            return getGain(table, attrIndex) / splitInfo;
        }

        double getInfoD(Table table) {
            double ret = 0.0;
            int itemCount = (int)table.data.size();
            map<string, int> labelCount;

            for (int i = 0; i < table.data.size(); i++) {
                labelCount[table.data[i].back()]++;
            }

            vector<pair<string, int>> labelVector(labelCount.begin(), labelCount.end());
            int numClasses = (int)labelVector.size();

            #pragma omp parallel for reduction(+:ret)
            for (int i = 0; i < numClasses; i++) {
                auto iter = labelVector[i]; 
                double p = (double)iter.second / itemCount;
                ret += -1.0 * p * log(p) / log(2);
            }

            return ret;
        }

        double getInfoAttrD(Table table, int attrIndex) {
            double ret = 0.0;
            int itemCount = (int)table.data.size();

            map<string, vector<int>> attrValueMap;
            for (int i = 0; i < table.data.size(); i++) {
                attrValueMap[table.data[i][attrIndex]].push_back(i);
            }

            vector<pair<string, vector<int>>> attrValueVector(attrValueMap.begin(), attrValueMap.end());
            int numValues = (int)attrValueVector.size();

       
            #pragma omp parallel for reduction(+:ret)
            for (int i = 0; i < numValues; i++) {
                auto iter = attrValueVector[i];

                Table nextTable; 
                for (int j = 0; j < iter.second.size(); j++) {
                    nextTable.data.push_back(table.data[iter.second[j]]);
                }
                int nextItemCount = (int)nextTable.data.size();
                ret += (double)nextItemCount / itemCount * getInfoD(nextTable);
            }

            return ret;
        }

        double getGain(Table table, int attrIndex) {
            return getInfoD(table) - getInfoAttrD(table, attrIndex);
        }

        double getSplitInfoAttrD(Table table, int attrIndex) {
            double ret = 0.0;
            int itemCount = (int)table.data.size();

            map<string, vector<int>> attrValueMap;
            for (int i = 0; i < table.data.size(); i++) {
                attrValueMap[table.data[i][attrIndex]].push_back(i);
            }
            
            vector<pair<string, vector<int>>> attrValueVector(attrValueMap.begin(), attrValueMap.end());
            int numValues = (int)attrValueVector.size();

            #pragma omp parallel for reduction(+:ret)
            for (int i = 0; i < numValues; i++) {
                auto iter = attrValueVector[i];
                int nextItemCount = (int)iter.second.size();

                double d = (double)nextItemCount / itemCount;
                if (d > 0.0) {
                    ret += -1.0 * d * log(d) / log(2);
                }
            }

            return ret;
        }


        // Exibe a estrutura da árvore
        void printTree(int nodeIndex, string branch) {
            if (tree[nodeIndex].isLeaf == true)
                cout << branch << "Label: " << tree[nodeIndex].label << "\n";

            for (int i = 0; i < tree[nodeIndex].children.size(); i++) {
                int childIndex = tree[nodeIndex].children[i];

                string attributeName = initialTable.attrName[tree[nodeIndex].criteriaAttrIndex];
                string attributeValue = tree[childIndex].attrValue;

                printTree(childIndex, branch + attributeName + " = " + attributeValue + ", ");
            }
        }
};

class InputReader {
    private:
        ifstream fin;
        Table table;
    public:
        InputReader(string filename) {
            fin.open(filename);
            if (!fin) {
                cout << filename << " file could not be opened\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            parse();
        }
        void parse() {
            string str;
            bool isAttrName = true;
            while (!getline(fin, str).eof()) {
                vector<string> row;
                int pre = 0;
                for (int i = 0; i < str.size(); i++) {
                    if (str[i] == '\t') {
                        string col = str.substr(pre, i - pre);
                        row.push_back(col);
                        pre = i + 1;
                    }
                }
                string col = str.substr(pre);
                if (!col.empty() && col.back() == '\r') {
                   col.pop_back();
                }
                row.push_back(col);


                if (isAttrName) {
                    table.attrName = row;
                    isAttrName = false;
                } else {
                    table.data.push_back(row);
                }
            }
        }
        Table getTable() {
            return table;
        }
};

class OutputPrinter {
    private:
        ofstream fout;
    public:
        OutputPrinter(string filename) {
            fout.open(filename);
            if (!fout) {
                cout << filename << " file could not be opened\n";
                MPI_Abort(MPI_COMM_WORLD, 1);  
            }
        }

        string joinByTab(vector<string> row) {
            string ret = "";
            for (int i = 0; i < row.size(); i++) {
                ret += row[i];
                if (i != row.size() - 1) {
                    ret += '\t';
                }
            }
            return ret;
        }

        void addLine(string str) {
            fout << str << endl;
        }
};



int main(int argc, const char* argv[]) {
    using namespace std::chrono; 

    MPI_Init((int*)&argc, (char***)&argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    omp_set_nested(1);

    if (argc != 4) {
        if (world_rank == 0) { 
            cout << "Please follow this format: mpirun -n [P] " << argv[0] << " [train.txt] [test.txt] [result.txt]";
        }
        MPI_Finalize();
        return 0;
    }

    high_resolution_clock::time_point start_total, end_total, start_train, end_train;


    if (world_rank == 0) {
        start_total = high_resolution_clock::now();
    }

    string trainFileName = argv[1];
    InputReader trainInputReader(trainFileName);

    if (world_rank == 0) {
        start_train = high_resolution_clock::now();
    }


    DecisionTree decisionTree(trainInputReader.getTable());

    if (world_rank == 0) {
        end_train = high_resolution_clock::now();
    }

    if (world_rank == 0) {
        string testFileName = argv[2];
        InputReader testInputReader(testFileName);
        Table test = testInputReader.getTable();

        string resultFileName = argv[3];
        OutputPrinter outputPrinter(resultFileName);
        outputPrinter.addLine(outputPrinter.joinByTab(test.attrName));

        for (int i = 0; i < test.data.size(); i++) {
            vector<string> result = test.data[i];
            result.push_back(decisionTree.guess(test.data[i]));
            outputPrinter.addLine(outputPrinter.joinByTab(result));
        }

        cout << endl << "=== Model Accuracy Evaluation ===" << endl;

        int correctTrain = 0;
        Table trainSet = trainInputReader.getTable();
        int totalTrain = (int)trainSet.data.size();

        for (int i = 0; i < trainSet.data.size(); i++) {
            string predicted = decisionTree.guess(trainSet.data[i]);
            string actual = trainSet.data[i].back();
            if (predicted == actual)
                correctTrain++;
        }

        cout << "Training accuracy: " << (double)correctTrain / totalTrain * 100.0
            << "% (" << correctTrain << "/" << totalTrain << ")" << endl;

        if (!test.attrName.empty()) {
            string lastAttr = test.attrName.back();
            lastAttr.erase(remove_if(lastAttr.begin(), lastAttr.end(), ::isspace), lastAttr.end());
            transform(lastAttr.begin(), lastAttr.end(), lastAttr.begin(), ::tolower);

            if (lastAttr == "class") {
                int correctTest = 0;
                int totalTest = (int)test.data.size();

                for (int i = 0; i < test.data.size(); i++) {
                    string predicted = decisionTree.guess(test.data[i]);
                    string actual = test.data[i].back();
                    if (predicted == actual)
                        correctTest++;
                }

                cout << "Test accuracy: " << (double)correctTest / totalTest * 100.0
                    << "% (" << correctTest << "/" << totalTest << ")" << endl;
            } else {
                cout << "Test file does not contain real class labels — skipping test accuracy." << endl;
            }
        }

        end_total = high_resolution_clock::now();
        double time_train = duration_cast<duration<double>>(end_train - start_train).count();
        double time_total = duration_cast<duration<double>>(end_total - start_total).count();

        cout << "=================================" << endl;
        cout << "Hybrid Execution (MPI + OpenMP)" << endl;
        cout << "MPI Processes: " << world_size << endl;
        cout << "OMP Nested Enabled: " << omp_get_nested() << endl;
        cout << "OMP Threads per Process: " << omp_get_max_threads() << " (control via OMP_NUM_THREADS)" << endl;
        cout << "---------------------------------" << endl;
        cout << "Training time: " << time_train << " seconds" << endl;
        cout << "Total execution time: " << time_total << " seconds" << endl;
        cout << "=================================" << endl;
    }

    MPI_Finalize();
    return 0;
}