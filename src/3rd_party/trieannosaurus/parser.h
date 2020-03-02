#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <iostream>
#if __has_include(<filesystem>)
# include <filesystem>
#else
# include <experimental/filesystem>
#endif
#include <filesystem>
#include <cmath>
#include <iomanip>
#ifndef _WIN32
#include <sys/ioctl.h> 
#endif

namespace trieannosaurus {

//Progress bar for file reading
//https://stackoverflow.com/questions/23400933/most-efficient-way-of-creating-a-progress-bar-while-reading-input-from-a-file
inline void ProgresBar(size_t cur_pos, size_t len, int barWidth) {
    float prog = std::min(cur_pos / float(len), 1.0f);
    int curPers = std::ceil(prog * barWidth);
    std::cerr << std::fixed << std::setprecision(2)
        << "\r   [" << std::string(curPers, '#')
        << std::string(barWidth + 1 - curPers, ' ') << "] " << 100 * prog << "%";
    if ((int)prog == 1) {
        std::cerr << std::endl;
    } else {
        std::cerr.flush();
    }
}

/*Adapted from https://www.bfilipek.com/2018/07/string-view-perf-followup.html . We should probably go string_view way*/
inline void tokenizeSentence(std::string& str, std::vector<std::string>& output, bool addEoS=false,
 std::string delimeter = " ") {
    auto first = std::begin(str);

    while (first != str.end()) {
        const auto second = std::find_first_of(first, std::end(str), std::begin(delimeter), std::end(delimeter));

        if (first != second) {
            output.emplace_back(str.substr(std::distance(std::begin(str), first), std::distance(first, second)));
        }

        if (second == str.end())
            break;

        first = std::next(second);
    }
    if (addEoS) {
        output.emplace_back("</s>");
    }
} 


class MakeVocab {
private:
    std::unordered_map<std::string, uint16_t> inmap;
    std::unordered_map<uint16_t, std::string> outmap;
    uint16_t vID = 0;
public:
    void operator()(std::string& line) {
        std::vector<std::string> tokens;
        tokenizeSentence(line, tokens, true);
        for (auto&& item : tokens) {
            if (inmap.find(item) == inmap.end()) {
                outmap.insert({vID, item});
                inmap.insert({item, vID});
                vID++;
            }
        }
    }
    std::pair<std::unordered_map<std::string, uint16_t>, std::unordered_map<uint16_t, std::string>> getMaps() {
        return {inmap, outmap};
    }
};

template <class StringType, class Operation>
void readFileByLine(StringType filename, Operation& op, const char * msg="") {
    //Get Terminal length on Linux
    int width_terminal = 80;
    #ifndef _WIN32
        struct winsize w;
        ioctl(0, TIOCGWINSZ, &w);
        width_terminal = std::ceil(w.ws_col*3/4);
    #endif
    size_t length;
    std::ifstream input;
    input.exceptions ( std::ifstream::badbit );
    try {
        input.open(filename);
        if (!input.good()) {
            std::cerr << "No such file or directory: " << filename << std::endl;
            std::exit(1);
        }
        input.seekg (0, input.end);
        length = input.tellg();
        input.seekg (0, input.beg);

        std::string line;
        std::cerr << "\r   " << msg << std::endl;
        while (getline(input, line)) {
            ProgresBar(input.tellg(), length, width_terminal);
            op(line);
        }
        input.close();
    } catch (const std::ifstream::failure& e) {
        std::cerr << "Error opening file " << filename << " : " << e.what() << std::endl;
        std::exit(1);
    }

}

} //Namespace
