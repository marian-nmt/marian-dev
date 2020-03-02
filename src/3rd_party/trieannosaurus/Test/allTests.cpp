#include "trieMe.h"
#define CATCH_CONFIG_MAIN 
#include "3rd_party/catch.hpp" 

namespace trieannosaurus {

TEST_CASE("Vocab", "[factorial]") {
    MakeVocab vocabFunctor;
    readFileByLine("Test/test_sents", vocabFunctor);
    std::unordered_map<std::string, uint16_t> inmap;
    std::unordered_map<uint16_t, std::string> outmap;
    auto maps = vocabFunctor.getMaps();
    inmap = maps.first;
    outmap = maps.second;
    {
        INFO("i is indexed as " << inmap.at("i") << " expected 0.")
        CHECK(inmap.at("i") == 0);
        CHECK(outmap.at(0) == "i");
    }
    {
        INFO("am is indexed as " << inmap.at("am") << " expected 1.")
        CHECK(inmap.at("am") == 1);
        CHECK(outmap.at(1) == "am");
    }
    {
        INFO("read is indexed as " << inmap.at("read") << " expected 8.")
        CHECK(inmap.at("read") == 8);
        CHECK(outmap.at(8) == "read");
    }
    {
        INFO(". is indexed as " << inmap.at(".") << " expected 11.")
        CHECK(inmap.at(".") == 11);
        CHECK(outmap.at(11) == ".");
    }
    {
        INFO("working is indexed as " << inmap.at("working") << " expected 25.")
        CHECK(inmap.at("working") == 25);
        CHECK(outmap.at(25) == "working");
    }
    {
        INFO("wrong is indexed as " << inmap.at("wrong") << " expected 28.")
        CHECK(inmap.at("wrong") == 28);
        CHECK(outmap.at(28) == "wrong");
    }
    {
        INFO("wrong is indexed as " << inmap.at("</s>") << " expected 4.")
        CHECK(inmap.at("</s>") == 4);
        CHECK(outmap.at(4) == "</s>");
    }

}

TEST_CASE("Trie", "[trie]") {
    MakeVocab vocab;
    readFileByLine("Test/test_sents", vocab);
    auto maps = vocab.getMaps();

    trieMeARiver trie(maps.first, maps.second);
    readFileByLine("Test/test_sents", trie);
    
    {
        std::string res = trie.find("i am");
        INFO("Expected \"very\", got " << res)
        CHECK(res == "very");
    }
    {
        std::string res = trie.find("this");
        INFO("Expected \"code pleasure\", got " << res)
        CHECK(res == "code pleasure");
    }
    {
        std::string res = trie.find("we need to read in some");
        INFO("Expected \"duplicates books\", got " << res)
        CHECK(res == "duplicates books");
    }
    {
        std::string res = trie.find("we need to read in some goose");
        INFO("Expected \"No continuations found\", got " << res)
        CHECK(res == "No continuations found");
    }
    {
        std::string res = trie.find("you have to </s>");
        INFO("Expected \"No continuations found\", got " << res)
        CHECK(res == "No continuations found");
    }
    {
        std::string res = trie.find("you have to");
        INFO("Expected \"</s>\", got " << res)
        CHECK(res == "</s>");
    }
    {
        std::string res = trie.find("we need to read in some duplicates ,");
        INFO("Expected \"so\", got " << res)
        CHECK(res == "so");
    }
    {
        std::string res = trie.find("please don't");
        INFO("Expected \"have beg hate fake\", got " << res)
        CHECK(res == "have beg hate fake");
    }
}
} //namespace
