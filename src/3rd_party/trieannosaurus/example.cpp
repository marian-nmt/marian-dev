#include "trieMe.h"

int main(int argc, char** argv) {
    using namespace trieannosaurus;
    std::string filename = "Test/test_sents";
    if (argc == 2) {
        filename = argv[1];
    }

    MakeVocab vocab;
    readFileByLine(filename, vocab, "Building vocabulary...");
    auto maps = vocab.getMaps();

    trieMeARiver trie(maps.first, maps.second);
    readFileByLine(filename, trie, "Building trie...");
    std::cout << trie.find("i am") << std::endl;
    std::cout << trie.find("this") << std::endl;
    std::cout << trie.find("we need to read in some") << std::endl;
    std::cout << trie.find("we need to read in some duplicates ,") << std::endl;
    std::cout << trie.find("please don't") << std::endl;
    std::cout << trie.find("you have to") << std::endl;
}
