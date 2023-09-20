#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#define adjweights(r, c) adjweights[r*n+c]

using namespace std;

unordered_set<string> stopwords = {
    "most", "wouldn't", "he", "whom", "does", "didn't", "again", "into", "needn't", "am", "both", "that'll", "aren't", "shouldn't", "now", "is",
    "ma", "wasn't", "hasn't", "under", "here", "ain", "between", "don", "while", "couldn", "with", "there", "be", "until", "ours", "will", "such",
    "him", "mustn't", "wasn't", "can", "herself", "weren", "should've", "hadn't", "of", "had", "shan", "ve", "which", "re", "my", "when",
    "because", "himself", "through", "then", "couldn't", "few", "to", "just", "o", "you", "so", "don't", "yourself", "the", "but", "me", "you'd",
    "from", "she's", "it's", "do", "yourselves", "out", "below", "ll", "on", "your", "she", "how", "and", "won't", "hadn", "hers", "didn't", "themselves",
    "doesn", "been", "were", "at", "d", "over", "a", "more", "who", "should", "his", "isn't", "up", "before", "some", "any", "off", "have", "own", "our",
    "we", "not", "doesn't", "an", "aren", "mightn", "down", "other", "these", "it", "each", "i", "during", "if", "myself", "mustn", "they", "same", "them",
    "very", "wouldn't", "theirs", "weren't", "nor", "haven", "needn", "having", "shan't", "doing", "all", "than", "as", "by", "no", "only", "their", "that",
    "shouldn", "against", "won", "its", "her", "m", "did", "y", "mightn't", "you'll", "above", "are", "being", "itself", "once", "s", "you've", "was", "what",
    "yours", "haven't", "further", "after", "where", "this", "ourselves", "about", "you're", "for", "those", "hasn't", "has", "or", "why", "too", "in", "isn't"
};

bool isEndofSentence(string s)
{
    char last = s[s.length() - 1];
    if (last == '.' || last == '!' || last == '?')
        return true;
    return false;
}

void read_sentences(
    const char * const filename,
    int * const np,
    int * const wordCountp,
    vector<string> ** const originalp,
    char ** const endingsp,
    int ** const lengthsp
)
{
    int n = 0;
    int wordCount = 0;
    int size = 5;
    vector<string> * original;
    char * endings;
    int * lengths;
    string tmp;
    ifstream document;

    document.open(filename);
    assert(document);

    original = new vector<string>;
    assert(original);
    endings = (char *) malloc(size * sizeof(*endings));
    assert(endings);
    lengths = (int *) calloc(size, sizeof(*lengths));
    assert(lengths);

    // read from the file
    while (document >> tmp) {
        ++wordCount;
        ++lengths[n];
        if (original->size() <= (unsigned int)n)
            original->push_back("");
        // put the sentence to original
        if ((*original)[n].size() == 0) {
            (*original)[n] += tmp;
        } else {
            (*original)[n] += ' ';
            (*original)[n] += tmp;
        }

        // check whether meet sentence ending
        if (isEndofSentence(tmp)) {
            endings[n] = tmp[tmp.length() - 1];
            (*original)[n].pop_back();
            ++n;
        }

        // expand the array as needed
        if (n == size) {
            size *= 2;
            endings = (char*) realloc(endings, size * sizeof(*endings));
            assert(endings);
            lengths = (int *) realloc(lengths, size * sizeof(*lengths));
            assert(lengths);
        }
    }
    document.close();

    // resize the array to minimize space cost
    endings = (char*) realloc(endings, n * sizeof(*endings));
    assert (endings);
    lengths = (int *) realloc(lengths, n * sizeof(*lengths));
    assert(lengths);

    // assign values to the pointer
    *originalp = original;
    *wordCountp = wordCount;
    *np = n;
    *endingsp = endings;
    *lengthsp = lengths;
}

void parse_sentences(
    int n,
    vector<string> * const original,
    vector<vector<string>*> ** const contentp
)
{
    vector<vector<string>*> * content = new vector<vector<string>*>;

    for(int i = 0; i < n; ++i) {
        content->push_back(new vector<string>);
    }
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < n; ++i) {
        string s;
        stringstream ss((*original)[i]);
        while (ss >> s) {
            for (unsigned int j = 0; j < s.size(); ++j)
                s[j] = tolower(s[j]);

            if (stopwords.find(s) != stopwords.end())
                continue;

            (*content)[i] -> push_back(s);
        }
    }

    *contentp = content;
}

void build_weights(
    int n,
    vector<vector<string>*> * content,
    float ** adjweightsp,
    float ** weightsumsp
)
{
    float * adjweights;
    float * weightsums;
    vector<string> *a, *b;

    adjweights = (float*)calloc(n * n, sizeof(*adjweights));
    assert(adjweights);
    weightsums = (float*)calloc(n, sizeof(*weightsums));
    assert(weightsums);

    #pragma omp parrallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            a = (*content)[i];
            b = (*content)[j];
            int count = 0;
            //#pragma omp parallel for collapse(2) reduction(+:count)
            for (int k = 0; k < a -> size(); ++k) {
                for (int l = 0; l < b -> size(); ++l) {
                    if ((*a)[k] == (*b)[l])
                        count++;
                }
            }
            float weight = (float)count / (float)(a->size() * b->size());
            adjweights(i, j) = weight;
            adjweights(j, i) = weight;
            weightsums[i] += weight;
            weightsums[j] += weight;
        }
    }

    *adjweightsp = adjweights;
    *weightsumsp = weightsums;
}

void calc_score(
    int n,
    float * adjweights,
    float * weightsums,
    float ** scoresp
)
{
    float th = 0.0001;
    float beta = 0.875;
    float prev = 1.0;
    float current = 100000.0;
    float * scores;
    float tmp;

    scores = (float*)malloc(n * sizeof(*scores));

    for (int i = 0; i < n; ++i) {
        scores[i] = 0.5;
    }

    while (abs(current - prev) > th) {
        for (int i = 0; i < n; ++i) {
            tmp = 0;
            #pragma omp parallel for reduction(+:tmp)
            for (int j = 0; j < n; ++j) {
                if (weightsums[j] != 0) {
                    tmp += scores[j] * adjweights(j, i) / weightsums[j];
                }
            }
            scores[i] = 1 - beta + beta * tmp;
        }
        prev = current;
        current = scores[0];
    }
    *scoresp = scores;
}

bool cmpPair(pair<float,int> p1, pair<float, int> p2) {
    return p1.first > p2.first ;
}
void print_top(
    int n,
    int words,
    float * scores,
    vector<string> * original,
    char * endings,
    int * lengths
)
{
    int count = 0;
    int index;
    vector<pair<float, int>> list;
    for (int i = 0; i < n; ++i) {
        list.push_back(pair<float, int>(scores[i], i));
    }
    sort(list.begin(), list.end(), cmpPair);

    for (int i = 0; i < n; ++i) {
        index = list[i].second;
        cout << (*original)[index] << endings[index] << endl;
        count += lengths[index];
        if (count > words) {
            return;
        }
    }

}

int main(int argc, char * argv[])
{
    vector<vector<string>*> * content;
    vector<string> * original;
    char * endings;
    int * lengths;
    float * adjweights;
    float * weightsums;
    float * scores;
    int n, wordCount;
    double ts, te;
    int nthreads = 2;

    if (argc < 2) {
        cout << "Invalid number of arguements." << endl;
        cout << "Usage: P1 <inputfile>" << endl;
        return EXIT_FAILURE;
    }

    if (argc >= 3) {
        nthreads = atoi(argv[2]);
        cout << "using " << nthreads << " threads" << endl;
    }
    omp_set_num_threads(nthreads);


    read_sentences(argv[1], &n, &wordCount, &original, &endings, &lengths);
    cout << "working on passage with " << n << " sentences and "
        << wordCount << " words" << endl;
    ts = omp_get_wtime();
    parse_sentences(n, original, &content);

    build_weights(n, content, &adjweights, &weightsums);

    calc_score(n, adjweights, weightsums, &scores);

    te = omp_get_wtime();
    cout << "calculate time: " << te - ts << endl;

    //print_top(n, wordCount / 5, scores, original, endings, lengths);

    for (int i = 0; i < n; ++i) {
        delete (*content)[i];
    }

    delete content;
    delete original;
    free(endings);
    free(lengths);
    free(adjweights);
    free(weightsums);
    free(scores);
}
